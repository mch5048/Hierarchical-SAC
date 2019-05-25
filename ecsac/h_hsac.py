#!/usr/bin/env python
import numpy as np
import tensorflow as tf
# import gym
import time as timer
import time
import hhsac_core
from hhsac_core import get_vars
from utils.logx import EpochLogger
from utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from Sawyer_PickAndPlaceEnv_v0 import robotEnv # REFACTOR TO REMOVE GOALS
import rospy
# Leveraging demonstration is same as TD3


from std_srvs.srv import Empty, EmptyRequest
from running_mean_std import RunningMeanStd
from SetupSummary import SummaryManager_HIRO as SummaryManager
import pickle
import os
# apply hindsight experience replay.
from her import Her_sampler
from OU import OU
from Gaussian import Gaussian
from collections import OrderedDict
from state_action_space_h_hsac import *
from tqdm import tqdm
from wandb.tensorflow import wandb



rms_path = '/home/irobot/catkin_ws/src/ddpg/scripts/ecsac_exp/'
log_path = '/home/irobot/catkin_ws/src/ddpg/scripts/ecsac_exp/'
# manager weight save dirs
MAN_PI_MODEL_SAVEDIR = '/home/irobot/catkin_ws/src/ddpg/scripts/ecsac_exp/man_pi_trained/'
MAN_QF1_MODEL_SAVEDIR = '/home/irobot/catkin_ws/src/ddpg/scripts/ecsac_exp/man_qf1_trained/'
MAN_QF2_MODEL_SAVEDIR = '/home/irobot/catkin_ws/src/ddpg/scripts/ecsac_exp/man_qf2_trained/'

# controller weight save dirs
CTRL_PI_MODEL_LOADDIR = '/home/irobot/catkin_ws/src/ddpg/scripts/ecsac_exp/ctrl_pi_trained/'
CTRL_VF_MODEL_LOADDIR = '/home/irobot/catkin_ws/src/ddpg/scripts/ecsac_exp/ctrl_vf_trained/'
CTRL_QF1_MODEL_LOADDIR = '/home/irobot/catkin_ws/src/ddpg/scripts/ecsac_exp/ctrl_qf1_trained/'
CTRL_QF2_MODEL_LOADDIR = '/home/irobot/catkin_ws/src/ddpg/scripts/ecsac_exp/ctrl_qf2_trained/'

# we exploit off-policy demo data with replay buffer
rms_path = '/home/irobot/catkin_ws/src/ddpg/scripts/ecsac_aux/'
demo_path = '/home/irobot/catkin_ws/src/ddpg/scripts/ecsac_aux/'
catkin_path = '/home/irobot/catkin_ws/'
MAN_BUF_FNAME = 'demo_manager_buffer.bin'
CON_BUF_FNAME = 'demo_controller_buffer.bin'

# BC_MODEL_LOADDIR = '/home/irobot/catkin_ws/src/ddpg/scripts/sac_exp/bc_pretrain/weights/'

class ReplayBuffer(object):
    """
    A simple FIFO experience replay buffer for ECSAC + HIRO agents.
    For HIRO, we need seperate replay buffers for both manager and controller policies.
    low-level controller does not require state-action sequence
    high-level controller requires extra buffer for state-action 
    """

    def __init__(self, obs_dim, meas_stt_dim, act_dim, aux_stt_dim, size, manager=False):
        # 9 types of data in buffer
        self.obs_buf = np.zeros(shape=(size,)+ obs_dim, dtype=np.float32) # o_t, (-1,100,100,3)
        self.obs1_buf = np.zeros(shape=(size,)+ obs_dim, dtype=np.float32) # o_t+1, (-1,100,100,3)
        self.g_buf = np.zeros(shape=(size , meas_stt_dim), dtype=np.float32) # g_t, (-1,21)
        self.g1_buf = np.zeros(shape=(size , meas_stt_dim), dtype=np.float32) # g_t+1, (-1,21)
        self.stt_buf = np.zeros(shape=(size , meas_stt_dim), dtype=np.float32) # s_t (-1,21), joint pos, vel, eff
        self.stt1_buf = np.zeros(shape=(size , meas_stt_dim), dtype=np.float32) # s_t+1 for (-1,21), consists of pos, vel, eff       self.act_buf = np.zeros(shape=(size, act_dim), dtype=np.float32) # A+t 3 dim for action
        if not manager:
            self.act_buf = np.zeros(shape=(size , act_dim), dtype=np.float32) # a_t (-1,8)
        else:
            self.act_buf = np.zeros(shape=(size , meas_stt_dim), dtype=np.float32)
        self.rews_buf = np.zeros(shape=(size,), dtype=np.float32)
        self.done_buf = np.zeros(shape=(size,), dtype=np.float32)
        self.aux_buf = np.zeros(shape=(size , aux_stt_dim), dtype=np.float32)
        self.aux1_buf = np.zeros( shape=(size , aux_stt_dim), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, obs1, g, g1, stt, stt1, act, aux, aux1, rew, done, manager=False):
        """store step transition in the buffer
        """
        self.obs_buf[self.ptr] = obs
        self.obs1_buf[self.ptr] = obs1
        self.g_buf[self.ptr] = g
        self.g1_buf[self.ptr] = g1
        self.stt_buf[self.ptr] = stt
        self.stt1_buf[self.ptr] = stt1
        self.act_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.aux_buf[self.ptr] = aux
        self.aux1_buf[self.ptr] = aux
        if not manager:
            self.ptr = (self.ptr+1) % self.max_size
            self.size = min(self.size+1, self.max_size)


    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        # 13 types of data in buffer
        return dict(ot=self.obs_buf[idxs],
                    ot1=self.obs1_buf[idxs],
                    g=self.g_buf[idxs],
                    g1=self.g1_buf[idxs],
                    st=self.stt_buf[idxs],
                    st1=self.stt1_buf[idxs],
                    acts=self.act_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs],
                    aux=self.aux_buf[idxs],
                    aux1=self.aux1_buf[idxs],
                    )


    def store_demo_transition(self, demo_batch):
        """ Store the expert demonstartion trajectory generated from DAgger
            demo_batch -> Tuple('data', 
            'size'= > should increase the ptr as this much.)
        """
        demo_size = demo_batch['size']
        self.obs_buf[:demo_size], self.obs1_buf[:demo_size], self.g_buf[:demo_size], \
        self.g1_buf[:demo_size], self.stt_buf[:demo_size], self.stt1_buf[:demo_size], \
        self.act_buf[:demo_size], self.rews_buf[:demo_size], self.done_buf[:demo_size], \
        self.aux_buf[:demo_size], self.aux1_buf[:demo_size] = demo_batch['data']
        self.ptr = (demo_size + 1) % self.max_size
        self.size = min(demo_size + 1, self.max_size)


class ManagerReplayBuffer(ReplayBuffer):
    """
    A simple FIFO experience replay buffer for ECSAC + HIRO agents.
    For HIRO, we need seperate replay buffers for both manager and controller policies.
    low-level controller does not require state-action sequence
    high-level controller requires extra buffer for state-action 
    """

    def __init__(self, obs_dim, meas_stt_dim, act_dim, aux_stt_dim,  size, seq_len):
        """full-state/ color_observation sequence lists' shape[1] is +1 longer than that of 
        action seqence -> they are stored as s_t:t+c/o_t:t+c, while action is stored as a_t:t+c
        """

        super(ManagerReplayBuffer, self).__init__(obs_dim, meas_stt_dim, act_dim, aux_stt_dim, size, manager=True)

        self.meas_stt_seq_buf = np.zeros(shape=(size, seq_len+1, meas_stt_dim), dtype=np.float32) # s_t (-1, 10+1, 21), joint pos, vel, eff
        self.aux_stt_seq_buf = np.zeros(shape=(size, seq_len+1, aux_stt_dim), dtype=np.float32) # s_t (-1, 10+1, 21), joint pos, vel, eff
        self.obs_seq_buf = np.zeros(shape=(size, seq_len+1,)+ obs_dim, dtype=np.float32) # o_t, (-1, 10+1, 100, 100, 3)
        self.act_seq_buf = np.zeros(shape=(size, seq_len, act_dim), dtype=np.float32) # a_t (-1,10, 8)

    def store(self, meas_stt_seq, aux_stt_seq, obs_seq, act_seq, *args, **kwargs):
        """store step transition in the buffer
        """
        super(ManagerReplayBuffer, self).store(manager=True, *args, **kwargs)
        
        self.meas_stt_seq_buf[self.ptr] = np.array(meas_stt_seq)
        self.aux_stt_seq_buf[self.ptr] = np.array(aux_stt_seq)
        self.obs_seq_buf[self.ptr] = np.array(obs_seq)
        self.act_seq_buf[self.ptr] = np.array(act_seq)
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)


    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        # 13 types of data in buffer
        return dict(ot=self.obs_buf[idxs],
                    ot1=self.obs1_buf[idxs],
                    g=self.g_buf[idxs],
                    g1=self.g1_buf[idxs],
                    st=self.stt_buf[idxs],
                    st1=self.stt1_buf[idxs],
                    acts=self.act_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs],
                    aux=self.aux_buf[idxs],
                    aux1=self.aux1_buf[idxs],
                    meas_st_seq=self.meas_stt_seq_buf[idxs],
                    aux_st_seq=self.aux_stt_seq_buf[idxs],
                    ot_seq=self.obs_seq_buf[idxs],
                    at_seq=self.act_seq_buf[idxs])

    def store_demo_transition(self, demo_batch):
        """ Store the expert demonstartion trajectory generated from DAgger
            demo_batch -> Tuple('data', 
            'size'= > should increase the ptr as this much.)
        """
        demo_size = demo_batch['size']
        self.obs_buf[:demo_size], self.obs1_buf[:demo_size], self.g_buf[:demo_size], \
        self.g1_buf[:demo_size], self.stt_buf[:demo_size], self.stt1_buf[:demo_size], \
        self.act_buf[:demo_size], self.rews_buf[:demo_size], self.done_buf[:demo_size], \
        self.aux_buf[:demo_size], self.aux1_buf[:demo_size] = demo_batch['data']
        
        # store a batch of sequence demo data
        self.meas_stt_seq_buf[:demo_size], \
        self.aux_stt_seq_buf[:demo_size], \
        self.obs_seq_buf[:demo_size], \
        self.act_seq_buf[:demo_size] = demo_batch['seq_data']
        self.ptr = (demo_size + 1) % self.max_size
        self.size = min(demo_size + 1, self.max_size)
"""

Midified Soft Actor-Critic
SAC + Asym-AC + Leveraging demos
(With slight variations that bring it closer to TD3)

Added hierarchical learning + learnable temperature (alpha).

"""
def td_target(reward, discount, next_value):
    return reward + discount * next_value

''
def assert_shape(tensor, expected_shape):
    tensor_shape = tensor.shape.as_list()
    assert len(tensor_shape) == len(expected_shape)
    assert all([a == b for a, b in zip(tensor_shape, expected_shape)])


def randomize_world():

    # We first wait for the service for RandomEnvironment change to be ready
    # rospy.loginfo("Waiting for service /dynamic_world_service to be ready...")
    rospy.wait_for_service('/dynamic_world_service')
    # rospy.loginfo("Service /dynamic_world_service READY")
    dynamic_world_service_call = rospy.ServiceProxy('/dynamic_world_service', Empty)
    change_env_request = EmptyRequest()

    dynamic_world_service_call(change_env_request)

def colorize_world():

    # We first wait for the service for RandomEnvironment change to be ready
    # rospy.loginfo("Waiting for service /dynamic_world_service to be ready...")
    rospy.wait_for_service('/colorize_world_service')
    # rospy.loginfo("Service /dynamic_world_service READY")
    dynamic_world_service_call = rospy.ServiceProxy('/dynamic_world_service', Empty)
    change_env_request = EmptyRequest()

    dynamic_world_service_call(change_env_request)


def normalize(ndarray, stats):
    if stats is None:
        return ndarray

    return (ndarray - stats.mean) / stats.std


def sample_action(action_dim):
    low = -1.0
    high = 1.0
    # high = high if self.dtype.kind == 'f' else self.high.astype('int64') + 1
    return np.concatenate([np.random.uniform(low=low, high=high, size=action_dim-1).astype('float32'), np.random.randint(2, size=1)], axis=-1)

    
def ecsac(train_indicator, isReal=False,logger_kwargs=dict()):
    
    # hiro specific params
    manager_propose_freq = 10
    train_manager_freq= 10
    manager_train_start =50
    high_start_timesteps = 5e2 # timesteps for random high-policy
    low_start_timesteps = 0 # timesteps for random low-policy
    candidate_goals = 8

    # general hyper_params
    seed=0 
    # epoch : in terms of training nn
    steps_per_epoch=4000
    epochs=100
    save_freq = 2e3 # every 2000 steps (2 epochs)
    gamma=0.99 # for both hi & lo level policies
    polyak=0.995 # tau = 0.005
    lr=1e-3 #for both actor
    pi_lr=3e-4
    vf_lr=3e-4 # for all the vf, and qfs
    alp_lr=3e-4 # for adjusting temperature
    batch_size=100
    start_steps=5000
    max_ep_len=1000

    #manager td3
    noise_idx = 1 # 0: ou 1: normal
    noise_type = "normal" if noise_idx else "ou"
    noise_stddev = 0.25
    manager_noise = 0.1 # for target policy smoothing.
    manager_noise_clip = 0.3 # for target policy smoothing.
    man_replay_size = int(5e4) # Memory leakage?
    delayed_update_freq = 4
    #controller sac
    ctrl_replay_size = int(5e4) # Memory leakage?
    target_ent = 0.01 # learnable entropy
    reward_scale_lo = 1.0

    # coefs for nn ouput regularizers
    reg_param = {'lam_mean':1e-3, 'lam_std':1e-3}
    lam_preact = 0.0

    # high-level manager pre-train params
    # train high-level policy for its mlp can infer joint states
    # encoding ?? -> rather, shouldn't it be LSTM? g 


    # wandb
    os.chdir(demo_path)
    wandb.init(project="hierarchical-sac", tensorboard=True)
    wandb.config.mananger_propose_freq = manager_propose_freq
    wandb.config.train_manager_freq = train_manager_freq
    wandb.config.manager_noise = manager_noise
    wandb.config.manager_noise_clip = manager_noise_clip
    wandb.config.manager_train_start = manager_train_start
    wandb.config.high_start_timesteps = high_start_timesteps
    wandb.config.low_start_timesteps = low_start_timesteps
    wandb.config.candidate_goals = candidate_goals
    wandb.config.steps_per_epoch = steps_per_epoch
    wandb.config.ctrl_replay_size, wandb.config.man_replay_size = ctrl_replay_size, man_replay_size
    wandb.config.gamma = gamma 
    wandb.config.polyak = polyak 
    wandb.config.polyak = polyak 
    wandb.config.pi_lr = pi_lr 
    wandb.config.vf_lr = vf_lr 
    wandb.config.alp_lr = alp_lr

    # model save/load
    USE_DEMO = True
    PRETRAIN_MANAGER = True
    USE_PRETRAINED_MANAGER = False
    DATA_LOAD_STEP = 20000
    high_pretrain_steps = int(5E4) 
    high_pretrain_save_freq = int(1e4)

    IS_TRAIN = train_indicator
    USE_CARTESIAN = True
    USE_GRIPPER = True

    exp_name = 'hiro-ecsac-td3'

    logger = EpochLogger(output_dir=log_path,exp_name=exp_name, seed=seed)
    logger.save_config(locals())

    tf.set_random_seed(seed)
    np.random.seed(seed)

    # arguments for RobotEnv : isdagger, isPOMDP, isGripper, isReal
    env = robotEnv(max_steps=max_ep_len, isPOMDP=True, isGripper=USE_GRIPPER, isCartesian=USE_CARTESIAN, train_indicator=IS_TRAIN)

    rospy.loginfo("Evironment has been created")
    # instantiate actor-critic nets for controller
    controller_actor_critic = hhsac_core.cnn_controller_actor_critic
    rospy.loginfo("Creates actor-critic nets for low-level contorller")
    # instantiate actor-critic nets for manager
    manager_actor_critic = hhsac_core.mlp_manager_actor_critic
    rospy.loginfo("Creates actor-critic nets for high-level manager")
    
    # for low_level controller
    obs_dim = (100, 100, 3) # for actor in POMDP
    meas_stt_dim = 21 # full_state of the robot (joint positions, velocities, and efforts) + ee position
    act_dim = 7 # 7 joint vels and gripper position 
    aux_dim = 3 # target object's position
    # define joint vel_limits
    action_space = (-1.0, 1.0)
    ee_dim = 7 # 4 quaternion
    grip_dim = 1 # 1 dimension for the gripper position

    # for high_level controller
    des_goal_dim = 21 # (joint positions, velocities, and efforts) + ee position
    sub_goal_dim = 21 # (joint positions, velocities, and efforts) -> should be partial observation (pos, vel, eff)

    if USE_CARTESIAN: # append 7-dim
        aux_dim += ee_dim
        # des_goal_dim += ee_dim

    if USE_GRIPPER: # append 1-dim
        meas_stt_dim += grip_dim
        des_goal_dim += grip_dim
        sub_goal_dim += grip_dim
        act_dim += grip_dim
        
    # Inputs to computation graph -> placeholders for next goal state is necessary.
    with tf.name_scope('placeholders'):
        obs_ph = tf.placeholder(dtype=tf.float32, shape=(None,)+obs_dim) # o_t : low_level controller
        obs1_ph = tf.placeholder(dtype=tf.float32, shape=(None,)+obs_dim) # o_t+1 : low_level controller
        sg_ph = tf.placeholder(dtype=tf.float32, shape=(None, meas_stt_dim)) # g_t_tilde : low_level controller
        sg1_ph = tf.placeholder(dtype=tf.float32, shape=(None, meas_stt_dim)) # g_t+1_tilde : low_level controller
        stt_ph = tf.placeholder(dtype=tf.float32, shape=(None, meas_stt_dim)) # s_t : low_level critic, high_level actor-critic -> measured states
        stt1_ph = tf.placeholder(dtype=tf.float32, shape=(None, meas_stt_dim)) # s_t+1 : low_level critic, high_level actor-critic -> measured states
        act_ph = tf.placeholder(dtype=tf.float32, shape=(None, act_dim)) # a_t : for low_level a-c
        aux_ph = tf.placeholder(dtype=tf.float32, shape=(None, aux_dim)) # only fed into the critic -> auxiliary states (ee_pose + measured_states )
        aux1_ph = tf.placeholder(dtype=tf.float32, shape=(None, aux_dim)) # only fed into the critic -> auxiliary states
        rew_ph_lo = tf.placeholder(dtype=tf.float32, shape=(None)) # r_t(s,g,a,s') = -1*l2_norm(s+g-s') : given by high-level policy
        rew_ph_hi = tf.placeholder(dtype=tf.float32, shape=(None)) # R_t = sparse (1 if suc else 0): given by the env.
        dn_ph = tf.placeholder(dtype=tf.float32, shape=(None)) # if episode is done : given by env.
    rospy.loginfo("placeholder have been created")

    # her_sampler = Her_sampler(reward_func=env.compute_reward_from_goal)
    # Experience buffer for seperate controller and manager
    controller_buffer = ReplayBuffer(obs_dim=obs_dim, meas_stt_dim=meas_stt_dim, act_dim=act_dim, aux_stt_dim=aux_dim, size=ctrl_replay_size)
    manager_buffer = ManagerReplayBuffer(obs_dim=obs_dim, meas_stt_dim=meas_stt_dim, act_dim=act_dim, aux_stt_dim=aux_dim, size=man_replay_size, seq_len=manager_propose_freq)
    
    # training ops definition
    # define operations for training high-level policy : TD3
    with tf.variable_scope('manager'): # removed desired goal
        with tf.variable_scope('main'):                                                     
            mu_hi, q1_hi, q2_hi, q1_pi_hi, pi_reg_hi = manager_actor_critic(stt_ph, sg_ph, aux_ph, action_space=None) # meas_St, Sg, aux_St
        # target_policy network
        with tf.variable_scope('target'):
            mu_hi_targ, _, _, _, _ = manager_actor_critic(stt1_ph, sg_ph, aux1_ph, action_space=None)

        # target_Q networks
        with tf.variable_scope('target', reuse=True): # re use the variable of q1 and q2
            # Target policy smoothing, by adding clipped noise to target actions
            # y = r + gamma*(1-done)*Q(s',g_des,sg'=mu_hi(s',g_des))
            epsilon = tf.random_normal(tf.shape(mu_hi_targ), stddev=manager_noise)
            epsilon = tf.clip_by_value(epsilon, -manager_noise_clip, manager_noise_clip)
            act_hi_targ = mu_hi_targ + epsilon
            # act_hi_targ = tf.clip_by_value(act_hi_targ, action_space[0], action_space[1]) # equivalent to periodic subgoals
            act_hi_targ = tf.minimum(s_high, tf.maximum(s_low, act_hi_targ)) # equivalent to periodic subgoals
            # target Q-values, using action from target policy
            _, q1_targ_hi, q2_targ_hi, _, _ = manager_actor_critic(stt1_ph, act_hi_targ, aux1_ph, action_space=None)
        
        # Losses for TD3
        with tf.name_scope('pi_loss_hi'):
            pi_loss_hi = -tf.reduce_mean(q1_pi_hi)
            # l2_loss_pi_hi = tf.losses.get_regularization_loss()
            # pi_loss_hi += l2_loss_pi_hi # regularization loss for the actor
            pi_loss_hi += pi_reg_hi * reg_param['lam_mean'] # regularization loss for the actor
        with tf.name_scope('q_loss_hi'):
            min_q_targ_hi = tf.minimum(q1_targ_hi, q2_targ_hi) # tensor
            q_backup_hi = tf.stop_gradient(rew_ph_hi + gamma*(1-dn_ph)*min_q_targ_hi)
            q1_loss_hi = tf.losses.mean_squared_error(labels=q_backup_hi, predictions=q1_hi, weights=0.5, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
            q2_loss_hi = tf.losses.mean_squared_error(labels=q_backup_hi, predictions=q2_hi, weights=0.5, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
            q_loss_hi = q1_loss_hi + q2_loss_hi
            l2_loss__q_hi = tf.losses.get_regularization_loss()
            q_loss_hi += l2_loss__q_hi

        # define training ops
        with tf.name_scope('optimize'):
            pi_hi_optimizer = tf.train.AdamOptimizer(learning_rate=pi_lr, name='pi_hi_optimizer')
            q_hi_optimizer = tf.train.AdamOptimizer(learning_rate=vf_lr, name='q_hi_optimizer')
            rospy.loginfo("manager optimizers have been created")
            
            train_pi_hi_op = tf.contrib.layers.optimize_loss(
                pi_loss_hi, global_step=None, learning_rate=pi_lr, optimizer=pi_hi_optimizer, variables=get_vars('manager/main/pi'),
                increment_global_step=False, summaries=("gradients", "gradient_norm", "global_gradient_norm"), name='pi_hi_opt')

            with tf.control_dependencies([train_pi_hi_op]): # expreimental application on high-level policy's learning in TD3.
                train_q_hi_op = tf.contrib.layers.optimize_loss(
                    q_loss_hi, global_step=None, learning_rate=pi_lr, optimizer=q_hi_optimizer, variables=get_vars('manager/main/q'),
                    increment_global_step=False, summaries=("gradients", "gradient_norm", "global_gradient_norm"), name='q_hi_opt')

            with tf.control_dependencies([train_q_hi_op]): # train_qfs 
                with tf.name_scope('polyak_hi_update'):
                    target_update_hi = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                                            for v_main, v_targ in zip(get_vars('manager/main'), get_vars('manager/target'))])
        # create manager summaries
        man_pi_summary = tf.summary.merge_all(scope='manager/optimize/pi_hi_opt', name='manager_pi_sumamry')
        man_q_summary = tf.summary.merge_all(scope='manager/optimize/q_hi_opt', name='manager_q_sumamry')
        rospy.loginfo("computational ops for manager are instantiated")

    # define operations for training low-level policy : SAC
    with tf.variable_scope('controller'):
        with tf.variable_scope('main'):
            mu_lo, pi_lo, logp_pi_lo, q1_lo, q2_lo, q1_pi_lo, q2_pi_lo, _, _, reg_losses = controller_actor_critic(stt_ph, obs_ph, sg_ph, act_ph, aux_ph, action_space=None)
            log_alpha_lo = tf.get_variable(name='log_alpha', initializer=-1.0, dtype=np.float32)
            alpha_lo = tf.exp(log_alpha_lo) 

        # Target value network
        with tf.variable_scope('target'):
            # _, _, _, _, _, _, _, v_targ_lo, _, _  = controller_actor_critic(stt1_ph, obs1_ph, sg1_ph, act_ph, aux1_ph, action_space=None)
            _, _, _, _, _, _, q1_pi_lo_targ, q2_pi_lo_targ, _, _  = controller_actor_critic(stt1_ph, obs1_ph, sg1_ph, act_ph, aux1_ph, action_space=None)

        with tf.name_scope('pi_loss_lo'):
            pi_loss_lo = tf.reduce_mean(alpha_lo * logp_pi_lo - q1_pi_lo) # grad_ascent for E[q1_pi+ alpha*H] > maximize return && maximize entropy
            # pi_l2_loss_lo = tf.losses.get_regularization_loss()
            # pi_loss_lo += pi_l2_loss_lo
            pi_loss_lo += reg_losses['preact_reg']*reg_param['lam_mean']  # regularization losses for the actor
            pi_loss_lo += reg_losses['std_reg']*reg_param['lam_std']

        with tf.name_scope('q_loss_lo'):
            # the original v ftn is not trained by minimizing the MSBE -> learned by the connection between Q and V
            # Legacy : min_q_pi_lo = tf.minimum(q1_pi_lo, q2_pi_lo)
            # impl_targ_v_lo = min_q_pi_lo - alpha_lo * logp_pi_lo # implicit valfue ftn thru soft q-ftn
            min_q_pi_lo_targ = tf.minimum(q1_pi_lo_targ, q2_pi_lo_targ)
            impl_targ_v_lo = min_q_pi_lo_targ - alpha_lo * logp_pi_lo

            q_backup_lo = tf.stop_gradient(rew_ph_lo + gamma*(1-dn_ph)*impl_targ_v_lo) 
            q1_loss_lo = tf.losses.mean_squared_error(labels=q_backup_lo, predictions=q1_lo, weights=0.5, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
            q2_loss_lo = tf.losses.mean_squared_error(labels=q_backup_lo, predictions=q2_lo, weights=0.5, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
            q1_l2_loss_lo = tf.losses.get_regularization_loss()
            q2_l2_loss_lo = tf.losses.get_regularization_loss()
            q1_loss_lo += q1_l2_loss_lo
            q2_loss_lo += q2_l2_loss_lo

        with tf.name_scope('alpha_loss_lo'):
            alpha_loss_lo = tf.reduce_mean(-log_alpha_lo*tf.stop_gradient(logp_pi_lo + target_ent)) # -alpha * log_p_pi - alpha * target_ent

        with tf.name_scope('optimize'):
            pi_lo_optimizer = tf.train.AdamOptimizer(learning_rate=pi_lr, name='pi_lo_optimizer')
            q1_lo_optimizer = tf.train.AdamOptimizer(learning_rate=vf_lr, name='q1_lo_optimizer')
            q2_lo_optimizer = tf.train.AdamOptimizer(learning_rate=vf_lr, name='q2_lo_optimizer')
            alpha_lo_optimizer = tf.train.AdamOptimizer(learning_rate=alp_lr, name='alpha_lo_optimizer')
        
            # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)
            rospy.loginfo("controller optimizers have been created")
            # explicitly compute gradients for debugging
            # learnable temperature
            train_alpha_lo_op = alpha_lo_optimizer.minimize(alpha_loss_lo,var_list=[log_alpha_lo])
            train_pi_lo_op = tf.contrib.layers.optimize_loss(
                pi_loss_lo, global_step=None, learning_rate=pi_lr, optimizer=pi_lo_optimizer, variables=get_vars('controller/main/pi'),
                increment_global_step=False, summaries=("gradients", "gradient_norm", "global_gradient_norm"), name='pi_lo_opt')

            with tf.control_dependencies([train_pi_lo_op]): # training pi won't affect learning vf
                train_q1_lo_op = tf.contrib.layers.optimize_loss(
                q1_loss_lo, global_step=None, learning_rate=vf_lr, optimizer=q1_lo_optimizer, variables=get_vars('controller/main/q1'),
                increment_global_step=False, summaries=("gradients", "gradient_norm"), name='q1_lo_opt')
                train_q2_lo_op = tf.contrib.layers.optimize_loss(
                q2_loss_lo, global_step=None, learning_rate=vf_lr, optimizer=q2_lo_optimizer, variables=get_vars('controller/main/q2'),
                increment_global_step=False, summaries=("gradients", "gradient_norm"), name='q2_lo_opt')
                train_q_lo_op = tf.group(train_q1_lo_op, train_q2_lo_op)
            # Polyak averaging for target variables
            # (control flow because sess.run otherwise evaluates in nondeterministic order)
            with tf.control_dependencies([train_q_lo_op]): # train_qfs 
                target_update_lo = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                                        for v_main, v_targ in zip(get_vars('controller/main'), get_vars('controller/target'))])

        # create controller summaries
        ctrl_pi_summary = tf.summary.merge_all(scope='controller/optimize/pi_lo_opt')
        ctrl_q1_summary = tf.summary.merge_all(scope='controller/optimize/q1_lo_opt')
        ctrl_q2_summary = tf.summary.merge_all(scope='controller/optimize/q2_lo_opt')
        rospy.loginfo("computational ops for controller are instantiated")


    # merge ops for manager
    with tf.name_scope('wandb_log'):
        step_hi_q_ops = [tf.reduce_mean(q1_hi), tf.reduce_mean(q2_hi), q1_loss_hi, q2_loss_hi, q_loss_hi, 
                        train_q_hi_op]
        step_hi_pi_ops = [pi_loss_hi, train_pi_hi_op, target_update_hi]
        step_hi_ops = {'q_ops': step_hi_q_ops, 'pi_ops':step_hi_pi_ops}
    
        # merge ops for controller
        step_lo_ops = [pi_loss_lo, q1_loss_lo, q2_loss_lo, tf.reduce_mean(q1_lo), tf.reduce_mean(q2_lo), logp_pi_lo, 
        train_pi_lo_op, train_q_lo_op, target_update_lo]
        alpha_lo_op = [train_alpha_lo_op, alpha_lo] # monitor alpha_lo
        step_lo_ops +=alpha_lo_op

    # Initializing targets to match main variables
    with tf.name_scope('init_networks'):
        target_init_hi = tf.group([tf.assign(v_targ, v_main)
                                for v_main, v_targ in zip(get_vars('manager/main'), get_vars('manager/target'))])
        target_init_lo = tf.group([tf.assign(v_targ, v_main)
                                for v_main, v_targ in zip(get_vars('controller/main'), get_vars('controller/target'))])
        target_init_ops = [target_init_hi, target_init_lo]

    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess = tf.Session(config=config)
    saver = tf.train.Saver()

    rospy.loginfo("initializing global variables...")

    sess.run(tf.global_variables_initializer())
    sess.run(target_init_ops)

    summary_writer = tf.summary.FileWriter(log_path + 'summary/', sess.graph)
                    #   color obs      pos    vel   eff   grip.   e.e.  obj(aux)   
    # obs_shape_list = [(1,100,100,3), (1,7), (1,7), (1,7), (1,1), (1,7), (1,3)]
    obs_shape_list = [(100,100,3), (7), (7), (7), (1), (7), (3)]

    # setup statistics summary and normalizers 
    summary_manager = SummaryManager(sess=sess, obs_shape_list=obs_shape_list, summary_writer=summary_writer)
    # summary_manager.setup_state_summary()
    # summary_manager.setup_stat_summary()
    # summary_manager.setup_train_summary()
    # summary_manager.setup_sac_summary() # logs ppo specific values
    # should make it more specific to HIRO architecture

    def normalize_action(action, space, batch_size=1):
        """ relocate and rescale the aciton of controller policy (subgoal) to desirable values.    
            action_dim = (7, 1) joint_vels + gripper_position
        """
        low = list()
        high = list()
        mean = list()
        scale = list()        
        for key, value in space.items():
            for k, v in value.items():
                low.append(v['lo'])
                high.append(v['hi'])
                mean.append(v['mean'])
                scale.append((v['hi']-v['lo'])/2)
        low = np.array(low).reshape(1,-1)
        high = np.array(high).reshape(1,-1)
        mean = np.array(mean).reshape(1,-1)
        scale = np.array(scale).reshape(1,-1)
        action = action * scale + mean
        norm_action = np.clip(action, low, high)
        return norm_action

    def normalize_subgoal(action, space, batch_size=1, issubgoal=True):
        """ relocate and rescale the aciton of manager policy (subgoal) to desirable values.    
            subgoal_policy = tensor of shape (1, 29(full_stt + ee + gripper) )
            subgoal_dim = dict(ee_pos=ee_pos, ee_quat=ee_quat, ee_rpy=ee_rpy,
                        joint_p=joint_p, joint_v=joint_v, joint_e=joint_e) 
            subgoal_dim = (3, 4, 3, 7, 7, 7) total 29-dim
        """
        if issubgoal:
            low = list()
            high = list()
            mean = list()
            scale = list()        
            for key, value in space.items():
                for k, v in value.items():
                    low.append(v['lo'])
                    high.append(v['hi'])
                    mean.append(v['mean'])
                    scale.append((v['hi']-v['lo'])/2)
            # here, reloc is the mean
            low = np.array(low).reshape(1,-1)
            high = np.array(high).reshape(1,-1)
            mean = np.array(mean).reshape(1,-1)
            scale = np.array(scale).reshape(1,-1)
            action = action * scale + mean
            norm_action = np.clip(action, low, high)
        return norm_action

    def reloc_rescale_gripper(gripper):
        """ relocate & rescale action to the desirable gripper 
        """
        raise DeprecationWarning
        # return 0.02*gripper + 0.022

    def get_action(obs, sub_goal, deterministic=False): # True for test
        """ infer primitive action from low-level policy
            TODO : manage when the obs and sub_goal are batch of actions
            7 joint velocities 
            TODO : np.copy the observation
        """
        act_op = mu_lo if deterministic else pi_lo # returned from actor_critic function
        sub_goal, obs = normalize_observation(meas_stt=sub_goal.copy().reshape((-1,sub_goal_dim)), c_obs=obs.copy().reshape((-1,)+obs_dim)) # subgoal_dim = meas_dim
        # TODO: resolve shape mismatching issue
        return np.squeeze(sess.run(act_op, feed_dict={obs_ph: obs,
                                           sg_ph: sub_goal}))

    def get_subgoal(state):
        """ infer subgoal from high-level policy for every freq.
        The output of the higher-level actor net is scaled to an approximated range of high-level acitons.
        7 joint torques : , 7 joint velocities: , 7 joint_positions: , 3-ee positions: , 4-ee quaternions: , 1 gripper_position:
        TODO : np.copy the observation
        """
        state = normalize_observation(meas_stt=state.copy().reshape((-1, meas_stt_dim)))
        # des_goal = normalize_observation(full_stt=des_goal.copy().reshape((-1,des_goal_dim)))
        return np.squeeze(sess.run(mu_hi, feed_dict={stt_ph: state,    
                                          }))

    def train_low_level_controller(train_ops, buffer, ep_len=max_ep_len, batch_size=batch_size, discount=gamma, polyak=polyak, step=0):
        """ train low-level actor-critic for each episode's timesteps
        TODO: determine what arguments to be parsed for this function.
        ops for low-level controller
            SEQUENCE : 1. Q_ftn 2. policy 3. temperature
            train_value_lo_op : learns value ftns 
            train_pi_lo_op : learns actor of the EC-SAC
            train_alpha_lo_op : learn temperature (alpha) 
            necessary phs for low-level policy :
                stt_ph, obs_ph, sg_ph, stt1_ph, obs1_ph, sg1_ph, act_ph, rew_ph_lo, dn_ph        
        fed train_ops :
            [pi_loss_lo, q1_loss_lo, q2_loss_lo, tf.reduce_mean(q1_lo), tf.reduce_mean(q2_lo), logp_pi_lo, 
                train_pi_lo_op, train_q_lo_op, target_update_lo]
            + [train_alpha_lo_op, alpha_lo] # monitor alpha_lo
        TODO: check for the placeholders for train_alpha_op
        """
        rospy.logwarn("Now trains the low-level controller for %d timesteps", ep_len)
        controller_ops = train_ops
        _ctrl_buffer = buffer
        for itr in tqdm(range(ep_len)):
            cur_step = step - ep_len + itr
            batch = _ctrl_buffer.sample_batch(batch_size)
            ctrl_feed_dict = {obs_ph: normalize_observation(c_obs=batch['ot']),
                         obs1_ph: normalize_observation(c_obs=batch['ot1']),
                         stt_ph: normalize_observation(meas_stt=batch['st']),
                         stt1_ph: normalize_observation(meas_stt=batch['st1']),
                         act_ph: normalize_observation(act=batch['acts']),
                         sg_ph : normalize_observation(meas_stt=batch['g']),
                         sg1_ph : normalize_observation(meas_stt=batch['g1']),
                         aux_ph : normalize_observation(aux_stt=batch['aux']),
                         aux1_ph : normalize_observation(aux_stt=batch['aux1']),
                         rew_ph_lo: (batch['rews']),
                         dn_ph: batch['done'],
                        }
            # low_outs = sess.run(controller_ops + monitor_lo_ops, ctrl_feed_dict)
            low_outs = sess.run(controller_ops +[ctrl_pi_summary, ctrl_q1_summary, ctrl_q2_summary], ctrl_feed_dict)
            # logging
            if itr % 10 == 0:
                wandb.log({'policy_loss_lo': low_outs[0], 'q1_loss_lo': low_outs[1],
                            'q2_loss_lo': low_outs[2], 'q1_lo': low_outs[3], 
                            'q2_lo': low_outs[4], 'alpha': low_outs[10], 'global_step': cur_step})

        summary_writer.add_summary(low_outs[-3], step) # low-pi summary
        summary_writer.add_summary(low_outs[-2], step) # low-q1 summary
        summary_writer.add_summary(low_outs[-1], step) # low-q2 summary

    def off_policy_correction(subgoals, s_seq, o_seq, a_seq, candidate_goals=8, batch_size=100, discount=gamma, polyak=polyak):
        """ run off policy correction for state - action sequence (s_t:t+c-1, a_t:t+c-1)
        e.g. shape = (batch_size, seq, state_dim)
        we need additional o_seq to generate actions from low-level policy
        """
        rospy.loginfo("Now we apply off-policy correciton")
        _s_seq = s_seq
        _o_seq = o_seq
        _a_seq = a_seq
        _candidate_goals = candidate_goals
        _subgoals = subgoals # batch of g_t
        first_states = [s[0] for s in _s_seq] # s_t
        last_states = [s[-1] for s in _s_seq] # s_t+c note that  the length of state seq is c+1
        first_obs = [o[0] for o in _o_seq] # o_t
        last_obs = [o[-1] for o in _o_seq] # o_t+c
        diff_goal = (np.array(last_states)-np.array(first_states))[:,np.newaxis,:] # s_t
        original_goal = np.array(subgoals)[:, np.newaxis, :] 
        random_goals = np.random.normal(loc=diff_goal, size=(batch_size, _candidate_goals, original_goal.shape[-1])) # gaussian centered @ s_t+c - s_t
        # TODO : modify the random goal samping 190525
        # shape (batch_size, 10, subgoal_dim)
        candidates = np.concatenate([original_goal, diff_goal, random_goals], axis=1)

        # should define what is 'subgoal'?
        # off-policy correction here.
        # candidtate goals from argument.

        og_shape = original_goal.shape
        # loc 
        random_goals = np.random.normal(loc=diff_goal, size=(batch_size, candidate_goals, original_goal.shape[-1]) ) #loc = mean of distribution
        rg_shape = random_goals.shape 
        candidates = np.concatenate([original_goal, diff_goal, random_goals], axis=1)
        _s_seq = np.array(_s_seq)[:,:-1,:] # check if s_seq has one more transition than required
        _o_seq = np.array(_o_seq)[:,:-1,:] # check if o_seq has one more transition than required
        _a_seq = np.array(_a_seq)
        seq_len = len(_s_seq[0])
        # flatten along the seq-dim
        flat_batch_size = seq_len * batch_size
        action_dim = _a_seq[0][0].shape
        state_dim = _s_seq[0][0].shape # except for the auxiliary observation
        obs_dim = _o_seq[0][0].shape # since the shape is (batch, seq_len,) + shape
        ncands = candidates.shape[1] # 10          
        #     # for off_policy correction, we need 
        #     # s_t:t+c-1
        rollout_actions = _a_seq.reshape((flat_batch_size,) + action_dim) #pi_lo actions from rollout
        states = _s_seq.reshape((flat_batch_size,) + state_dim) 
        observations = _o_seq.reshape((flat_batch_size,) + obs_dim)
        # shape of the np.array 'observations' -> (flat_batch_size, 24)
        # shape of the np.array 'batched_candidiates[i]' -> (flat_batch_size, 24)
        batched_candidates = np.tile(candidates, [seq_len, 1, 1]) # candidate goals
        batched_candidates= batched_candidates.transpose(1,0,2)
        pi_lo_actions = np.zeros((ncands, flat_batch_size) + action_dim) # shape (10, flat_batch_size, 8)
        # for which goal the new low policy would have taken the same action as the old one?
        # TODO: debug the shape for the batch action estimations
        # (1000, 100, 100, 3)
        # (10, 29)

        # TODO : modify here with gaussian likelihood 190525
        # requirements : alpha(global), log_std of the current low level policy.
        # def gaussian_likelihood(x, mu, log_std):
        for c in range(ncands): # one sample for each iteration?
            pi_lo_actions[c] = get_action(observations, batched_candidates[c]) #320 action estimations for each candidate out of 10
        # now compute the approx. lob prob. of the policy : eq.(5) in the paper
        difference = pi_lo_actions - rollout_actions # broadcasted, (10, flat_batch_size, 8) 
        difference = difference.reshape((ncands, batch_size, seq_len)+ action_dim) # shape (10, batch_size, 10, 8)
        diff_tr = difference.transpose(1,0,2,3) # shape (batch_size, n_cands, seq_len, 8)
        diff_norm = np.linalg.norm(diff_tr, axis=-1) # shape (batch_size, n_cands, seq_len)-> ||ai - mu_lo(si,gi_tilde)||_2
        log_prob = -0.5*np.sum(diff_norm**2, axis=-1) # sum along the seq_len (10)
        max_indicies = np.argmax(log_prob, axis=-1) # argmax along the n_cands -> batch of goals having the most log_prob
        # shape of max_indices ->(batch_size,) -> index along 10 candidates for each batch
        # shape of the candidates (batch, n_cands, goal_dim)
        return candidates[np.arange(batch_size),max_indicies] # [for each batch, max index]

    def train_high_level_manager(train_ops, buffer, ep_len=(max_ep_len/manager_propose_freq), batch_size=batch_size, discount=gamma, polyak=polyak, step=0):
        """ train high-level actor-critic for each episode's (timesteps/manager_train_freq)
        TODO: determine what arguments to be parsed for this function.
        ops for high-level controller -> must in order for TD3 leanring.
            NOTE : actor of the td3 is updated in delayed manner -> hyper param.
            NOTE : off-policy correction must be done here.
            SEQUENCE : 1. 2. 3. 
        TODO: check if primitive actions are required for training high-level manager.
        fed train_ops :            
            [pi_loss_hi, q1_hi, q2_hi, q1_loss_hi, q2_loss_hi, q_loss_hi, 
                train_pi_hi_op, train_q_hi_op, target_update_hi]
            1. learn Q-ftn : y = r + gamma*(1-done)*Q(s'||g_des,sg'=mu_hi(s'||g_des))
            2. learn pi    : E[Q(s||g_des,mu_hi(s||g_des))]
                -> placeholder for sg_ph is only required for 'target Q tfn' ops.
        """
        rospy.logwarn("Now trains the high-level controller for %d timesteps", ep_len)
        _man_buffer = buffer
        # execute off-policy correction before training
        for itr in tqdm(range(ep_len)):
            cur_step = step - int(ep_len * manager_propose_freq) + itr
            batch = _man_buffer.sample_batch(batch_size)
            # subgoal here is the action of manager network, action for computing log-likelihood
            # shape of corr_subgoals (batch_size, goal_dim)
            corr_subgoals = off_policy_correction(subgoals= batch['acts'],s_seq= batch['meas_st_seq'],
            o_seq= batch['ot_seq'], a_seq= batch['at_seq'])
            # action of the manager is subgoal...
            man_feed_dict = {stt_ph: normalize_observation(meas_stt=batch['st']),
                            stt1_ph: normalize_observation(meas_stt=batch['st1']),
                            sg_ph: normalize_observation(meas_stt=corr_subgoals),
                            aux_ph : normalize_observation(aux_stt=batch['aux']),
                            aux1_ph : normalize_observation(aux_stt=batch['aux1']),
                            rew_ph_hi: batch['rews'],
                            dn_ph: batch['done'],
                            }  # remove desired goal placeholders
            # modify here! 
            q_ops = train_ops['q_ops'] # [q1_hi, q2_hi, q1_loss_hi, q2_loss_hi, q_loss_hi, train_q_hi_op]
            pi_ops = train_ops['pi_ops'] # [pi_loss_hi, train_pi_hi_op, target_update_hi]
            q_hi_outs = sess.run(q_ops+[man_q_summary], man_feed_dict)

            if itr % delayed_update_freq == 0: # delayed update of the policy and target nets.
                pi_hi_outs = sess.run(pi_ops + [man_pi_summary], man_feed_dict)
                summary_writer.add_summary(pi_hi_outs[-1] , cur_step) # low-pi summary
                wandb.log({'policy_loss_hi': pi_hi_outs[0]})       
        wandb.log({'q1_hi': q_hi_outs[0], 'q2_hi': q_hi_outs[1], 'q1_loss_hi': q_hi_outs[2],
                    'q2_loss_hi': q_hi_outs[3], 'q_loss_hi': q_hi_outs[4], 'global_step': step})
        rospy.loginfo('writes summary of high-level value-ftn')
        summary_writer.add_summary(q_hi_outs[-1] , step) # low-q summary

    def pretrain_manager(demo_buffer, pretrain_steps, train_ops, batch_size=batch_size):
        """ Pre-trains the manager actor-critic network with data collected from demonstartions.
        TODO: check if tensorboard logging is valid here. 
        """
        if USE_PRETRAINED_MANAGER:
            os.chdir(catkin_path)
            new_saver = tf.train.import_meta_graph(os.getcwd()+'/src/ddpg/scripts/ecsac/model/ecsac_pretrain.ckpt-{0}.meta'.format(DATA_LOAD_STEP))
            new_saver.restore(sess, os.getcwd()+'/src/ddpg/scripts/ecsac/model/ecsac_pretrain.ckpt-{0}'.format(DATA_LOAD_STEP))
        else:
            rospy.logwarn("Pre-trains the high-level controller for %d timesteps", pretrain_steps)
            _demo_buffer = demo_buffer
            # execute off-policy correction before training
            for itr in tqdm(range(pretrain_steps)):
                batch = _demo_buffer.sample_batch(batch_size)
                # off policy  correction is not required here 
                # action of the manager is subgoal...
                man_feed_dict = {
                                stt_ph : normalize_observation(meas_stt=batch['st']),
                                stt1_ph : normalize_observation(meas_stt=batch['st1']),
                                sg_ph : normalize_observation(meas_stt=batch['acts']),
                                aux_ph : normalize_observation(aux_stt=batch['aux']),
                                aux1_ph : normalize_observation(aux_stt=batch['aux1']),
                                rew_ph_hi: batch['rews'],
                                dn_ph : batch['done'],}                                                    
                q_ops = train_ops['q_ops'] # [q1_hi, q2_hi, q1_loss_hi, q2_loss_hi, q_loss_hi, train_q_hi_op]
                pi_ops = train_ops['pi_ops'] # [pi_loss_hi, train_pi_hi_op, target_update_hi]
                _ = sess.run(q_ops, man_feed_dict)
                if itr + 1 % delayed_update_freq == 0: # delayed update of the policy and target nets.
                    _ = sess.run(pi_ops, man_feed_dict)
                if itr + 1 % high_pretrain_save_freq == 0:
                    rospy.loginfo('##### saves manager_pretrain weights ##### for step %d', itr + 1)
                    saver.save(sess,os.getcwd()+'/src/ddpg/scripts/ecsac/model/ecsac_pretrain.ckpt', global_step=itr + 1)
                    saver.save(sess, os.path.join(wandb.run.dir, '/model/ecsac_pretrain.ckpt'), global_step=itr + 1)
            
    def normalize_observation(meas_stt=None, c_obs=None, aux_stt=None, act=None):
        """ normalizes observations for each step based on running mean-std.
        will only be used for each subgoal /action estimation from the manager/controller policy
        meas_stt : [qpos, qvel, effort, grip_pos], aux_stt : [obj_pos, ee_pose]
        """
        if c_obs is not None and meas_stt is None:
            c_obs[:] =  normalize(c_obs[:], summary_manager.s_t0_rms)
            return c_obs
        elif meas_stt is not None and c_obs is None:
            meas_stt[:,:7] = normalize(meas_stt[:,:7], summary_manager.s_t1_rms) # joint pos
            meas_stt[:,7:14] = normalize(meas_stt[:,7:14], summary_manager.s_t2_rms) # joint vel
            meas_stt[:,14:21] = normalize(meas_stt[:,14:21], summary_manager.s_t3_rms) # joint eff
            meas_stt[:,21:] = normalize(meas_stt[:,21:], summary_manager.s_t4_rms) # gripper position
            return meas_stt
        elif act is not None:
            act[:] = normalize(act[:], summary_manager.a_t_rms)
            return act
        elif aux_stt is not None:
            print (aux_stt.shape)
            aux_stt[:, :3] = normalize(aux_stt[:, :3], summary_manager.s_t6_rms) # obj_pos
            aux_stt[:, 3:] = normalize(aux_stt[:, 3:], summary_manager.s_t5_rms) # ee_pose 
            return aux_stt
        else:
            c_obs[:] = normalize(c_obs[:], summary_manager.s_t0_rms)
            meas_stt[:,:7] = normalize(meas_stt[:,:7], summary_manager.s_t1_rms) # joint pos
            meas_stt[:,7:14] = normalize(meas_stt[:,7:14], summary_manager.s_t2_rms) # joint vel
            meas_stt[:,14:21] = normalize(meas_stt[:,14:21], summary_manager.s_t3_rms) # joint eff
            meas_stt[:,21:] = normalize(meas_stt[:,21:], summary_manager.s_t4_rms) # gripper position
            return meas_stt, c_obs

    def update_rms(meas_stt=None, c_obs=None, aux_stt=None, act=None):
        """Update the mean/stddev of the running mean-std normalizers.
        Normalize full-state, color_obs, and auxiliary observation.
        Caution on the shape!
        """
        # TODO: modify here!
        summary_manager.s_t0_rms.update(c_obs) # c_obs
        summary_manager.s_t1_rms.update(meas_stt[:7]) # joint_pos
        summary_manager.s_t2_rms.update(meas_stt[7:14]) # joint_vel
        summary_manager.s_t3_rms.update(meas_stt[14:21]) # joint_eff
        summary_manager.s_t4_rms.update(meas_stt[21:]) # gripper_position
        summary_manager.s_t5_rms.update(aux_stt[3:]) # ee_pose
        summary_manager.s_t6_rms.update(aux_stt[:3]) # aux
        summary_manager.a_t_rms.update(act) # ee_pose

    def load_rms():
        rospy.logwarn('Loads the mean and stddev for test time')
        summary_manager.s_t0_rms.load_mean_std(rms_path+'mean_std0.bin')
        summary_manager.s_t1_rms.load_mean_std(rms_path+'mean_std1.bin')
        summary_manager.s_t2_rms.load_mean_std(rms_path+'mean_std2.bin')
        summary_manager.s_t3_rms.load_mean_std(rms_path+'mean_std3.bin')
        summary_manager.s_t4_rms.load_mean_std(rms_path+'mean_std4.bin')
        summary_manager.s_t5_rms.load_mean_std(rms_path+'mean_std5.bin')
        summary_manager.s_t6_rms.load_mean_std(rms_path+'mean_std6.bin')
        summary_manager.a_t_rms.load_mean_std(rms_path+'mean_std7.bin')

    def load_demo_rms():
        """ Loads the mean-stddev of the demonstration batch data
        """
        rospy.logwarn('Loads the mean and stddev for test time')
        summary_manager.s_t0_rms.load_mean_std(rms_path+'mean_std0_demo.bin')
        summary_manager.s_t1_rms.load_mean_std(rms_path+'mean_std1_demo.bin')
        summary_manager.s_t2_rms.load_mean_std(rms_path+'mean_std2_demo.bin')
        summary_manager.s_t3_rms.load_mean_std(rms_path+'mean_std3_demo.bin')
        summary_manager.s_t4_rms.load_mean_std(rms_path+'mean_std4_demo.bin')
        summary_manager.s_t5_rms.load_mean_std(rms_path+'mean_std5_demo.bin')
        summary_manager.s_t6_rms.load_mean_std(rms_path+'mean_std6_demo.bin')
        summary_manager.a_t_rms.load_mean_std(rms_path+'mean_std7_demo.bin')
    
    def save_rms(step):
        rospy.logwarn('Saves the mean and stddev @ step %d', step)
        summary_manager.s_t0_rms.save_mean_std(rms_path+'mean_std0.bin')
        summary_manager.s_t1_rms.save_mean_std(rms_path+'mean_std1.bin')
        summary_manager.s_t2_rms.save_mean_std(rms_path+'mean_std2.bin')
        summary_manager.s_t3_rms.save_mean_std(rms_path+'mean_std3.bin')
        summary_manager.s_t4_rms.save_mean_std(rms_path+'mean_std4.bin')
        summary_manager.s_t5_rms.save_mean_std(rms_path+'mean_std5.bin')
        summary_manager.s_t6_rms.save_mean_std(rms_path+'mean_std6.bin')
        summary_manager.a_t_rms.save_mean_std(rms_path+'mean_std7.bin')
    
    # TODO : implement high-level pre-train procedure here


    TRAIN_HIGH_LEVEL = False # DAgger should be reimplemented 

    episode_num = 0
    if TRAIN_HIGH_LEVEL: 
        raise NotImplementedError
        for tr in range(high_pretrain_steps) and not rospy.is_shutdown():
            obs = env.reset() # observation = {'meas_state': ,'auxiliary': ,'color_obs':}
            done = False
            reset = False
            ep_len = 0 # length of the episode
            ep_ret = 0 # episode return for the manager
            ep_low_ret = 0 # return of the intrinsic reward for low level controller 
            episode_num += 1 # for every env.reset()

    if USE_DEMO:
        with open(MAN_BUF_FNAME, 'rb') as f:
            rospy.logwarn('Loading demo batch on the manager buffer. May take a while...')
            _manager_batch = pickle.load(f)
            manager_buffer.store_demo_transition(_manager_batch)

        with open(CON_BUF_FNAME, 'rb') as f2:
            rospy.logwarn('Loading demo batch on the controller buffer. May take a while...')
            _controller_batch = pickle.load(f2)
            controller_buffer.store_demo_transition(_controller_batch)
        load_demo_rms()
        rospy.loginfo('Successfully loaded demo transitions on the buffers!')

    # back to catkin_ws to resolve directory conflict issue
    if PRETRAIN_MANAGER:
        pretrain_manager(demo_buffer=manager_buffer, pretrain_steps=high_pretrain_steps, train_ops=step_hi_ops)

    if noise_type == "normal":
        subgoal_noise = Gaussian(mu=np.zeros(sub_goal_dim), sigma=noise_stddev*np.ones(sub_goal_dim))
    else:
        subgoal_noise = OU(mu=np.zeros(sub_goal_dim), sigma=noise_stddev*np.ones(sub_goal_dim))

    start_time = time.time()
    ep_ret, ep_len = 0, 0
    total_steps = steps_per_epoch * epochs # total interaction steps in terms of training.
    t = 0 # counted steps [0:total_steps - 1]
    timesteps_since_manager = 0 # to count c-step elapse for training manager
    timesteps_since_subgoal = 0 # to count c-step elapse for subgoal proposal
    episode_num = 0 # incremental episode counter
    done = True
    reset = False
    manager_temp_transition = list() # temp manager transition
    if train_indicator: # train
        saver.save(sess,'/home/irobot/catkin_ws/src/ddpg/scripts/ecsac/model/ecsac.ckpt', global_step=t)
    else: # test
        rospy.logwarn('Now loads the pretrained weight for test')
        load_rms()    
        new_saver = tf.train.import_meta_graph(os.getcwd()+'/src/ddpg/scripts/ecsac/model/ecsac.ckpt-{0}.meta'.format(DATA_LOAD_STEP))
        new_saver.restore(sess, os.getcwd()+'/src/ddpg/scripts/ecsac/model/ecsac.ckpt-{0}'.format(DATA_LOAD_STEP))
    # Main loop: collect experience in env and update/log each epoch
    while not rospy.is_shutdown() and t < int(total_steps):

        if done or ep_len== max_ep_len: # if an episode is finished (no matter done==True)
            if t != 0 and train_indicator:
                # train_low level controllers for the length of this episode
                train_low_level_controller(train_ops=step_lo_ops, buffer=controller_buffer, ep_len=ep_len, step=t)
        
                # train high-level manager policy in delayed manner.
                if timesteps_since_manager >= train_manager_freq:
                    timesteps_since_manager = 0
                    train_high_level_manager(train_ops=step_hi_ops, buffer=manager_buffer, ep_len=int(ep_len/train_manager_freq), step=t)

                # Process final state/obs, store manager transition i.e. state/obs @ t+c
                if len(manager_temp_transition[3]) != 1: # if not terminal state -> no next state will be observed
                    #                               0       1    2      3    4     5    6     7     8     9     10    11     12    13    
                    # buffer store arguments :    s_seq, o_seq, a_seq, obs, obs_1, dg,  dg_1, stt, stt_1, act,  aux, aux_1,  rew, done 
                    manager_temp_transition[5] = c_obs # save o_t+c
                    manager_temp_transition[9] = meas_stt # save meas_stt_t+c
                    manager_temp_transition[12] = aux_stt # save aux_stt_t+c
                    manager_temp_transition[-1] = float(True) # done = True for manager, regardless of the episode
                # make sure every manager transition have the same length of sequence
                # TODO: debug here...
                if len(manager_temp_transition[0]) <= manager_propose_freq: # len(state_seq) = propose_freq +1 since we save s_t:t+c as a seq.
                    while len(manager_temp_transition[0]) <= manager_propose_freq:
                        manager_temp_transition[0].append(meas_stt) # s_seq
                        manager_temp_transition[1].append(aux_stt) # o_seq
                        manager_temp_transition[2].append(c_obs) # o_seq
                        manager_temp_transition[3].append(action) # a_seq
                # buffer store arguments : s_seq, o_seq, a_seq, obs, obs_1, dg, dg_1, stt, stt_1, act, aux, aux_1, rew, done 
                manager_buffer.store(*manager_temp_transition) # off policy correction is done inside the manager_buffer 

                save_rms(step=t)
            # reset the environment since an episode has been finished
            obs = env.reset() # observation = {'meas_state': ,'auxiliary': ,'color_obs':}
            done = False
            reset = False
            ep_len = 0 # length of the episode
            ep_ret = 0 # episode return for the manager
            ep_low_ret = 0 # return of the intrinsic reward for low level controller 
            episode_num += 1 # for every env.reset()

            # process observations
            meas_stt = np.concatenate(obs['observation']['meas_state'], axis=0) # s_meas
            c_obs = obs['observation']['color_obs'] #o_0
            des_goal = np.concatenate(obs['desired_goal']['full_state'], axis=0) # g_des
            aux_stt = np.concatenate(obs['observation']['auxiliary'], axis=0) # s_aux
            full_stt = np.concatenate([meas_stt, aux_stt], axis=0) 

            # infer subgoal for low-level policy
            subgoal = get_subgoal(meas_stt) # sub_goal_dim = (1, meas_stt_dim)
            _subgoal_noise = subgoal_noise() # random sample when called
            subgoal += _subgoal_noise
            timesteps_since_subgoal = 0
            # apply noise on the subgoal
            # create a temporal high-level transition : requires off-policy correction
            # buffer store arguments :    s_seq,    o_seq, a_seq, obs_t,  obs_t+c, dg, dg_t+c, meas_stt_t, meas_stt_t+c, subgoal, aux_stt_t, aux_stt_t+c,  rew, done  # check reward
            manager_temp_transition = [[meas_stt], [aux_stt], [c_obs], [], c_obs, None, None, None, meas_stt, None, subgoal, aux_stt, None,  0, False] # TODO: no reward @ init 
                                    #       0           1        2      3   4       5     6     7       8       9   10        11      12    13  14   
        if t < low_start_timesteps:
            action = sample_action(act_dim) # a_t
        else:
            action = get_action(c_obs, subgoal, deterministic= not train_indicator) # a_t
            # TODO: make action on the gripper as categorical policy
            # action[-1] = reloc_rescale_gripper(action[-1])
        ep_len += 1 # ep_len should be here!
        
        next_obs, manager_reward, done = env.step(action, time_step=ep_len) # reward R_t-> for high-level manager -> for sum(R_t:t+c-1)
        if train_indicator:
            randomize_world()
        # update episodic logs
        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state) -> if done = False for the max_timestep
        # DO NOT Make done = True when it hits timeout
        ep_ret += manager_reward # reward in terms of achieving episodic task
        done = False if ep_len== max_ep_len else done
        if done:
            rospy.logwarn('=============== Now epsiode %d ends with done signal! ====================', episode_num)
        next_meas_stt = np.concatenate(next_obs['observation']['full_state'], axis=0) # s_t
        next_c_obs = next_obs['observation']['color_obs'] #o_t
        next_aux_stt = np.concatenate(next_obs['observation']['auxiliary'], axis=0) # s_aux
        next_full_stt = np.concatenate([next_meas_stt, next_aux_stt], axis=0) 

        if train_indicator:
            # append manager transition
            manager_temp_transition[-1] = done
            manager_temp_transition[-2] += manager_reward # sum(R_t:t+c)
            manager_temp_transition[0].append(next_meas_stt) # append s_meas_seq
            manager_temp_transition[1].append(next_aux_stt) # append s_aux_seq
            manager_temp_transition[2].append(next_c_obs) # append o_seq
            manager_temp_transition[3].append(action)     # append a_seq        
            # compute intrinsic reward
            intrinsic_reward = env.compute_intrinsic_reward(meas_stt, next_meas_stt, subgoal)


        # subgoal transition
        next_subgoal = env.env_goal_transition(meas_stt, next_meas_stt, subgoal)
        # add transition for low-level policy
        # (obs, obs1, sg, sg1, stt, stt1, act, aux, rew, done)
        if train_indicator:
            controller_buffer.store(c_obs, next_c_obs, subgoal, 
            next_subgoal, meas_stt, next_meas_stt, action, aux_stt, next_aux_stt, intrinsic_reward, done)
        
        # observation transiton -> Move MDP by one timestep
        obs = next_obs
        meas_stt = next_meas_stt
        aux_stt = next_aux_stt
        subgoal = next_subgoal
        full_stt = next_full_stt

        # update logging steps
        ep_len += 1
        t +=1
        timesteps_since_manager += 1
        timesteps_since_subgoal += 1

        if train_indicator:        
            if timesteps_since_subgoal % manager_propose_freq == 0:
                # for every c-step, renew the subgoal estimation from the manager policy.
                # idx: [[s_meas], [s_aux], [c_obs], [act],  ]
                timesteps_since_subgoal = 0
                manager_temp_transition[5] = c_obs # save o_t+c
                manager_temp_transition[9] = meas_stt # save s_t+c
                manager_temp_transition[11] = aux_stt # save s_t+c
                manager_temp_transition[-1] = float(True) # done = True for manager, regardless of the episode 
                
                # intentional seq appending is not required here since it always satisfies c step.
                manager_buffer.store(*manager_temp_transition)
                subgoal = get_subgoal(full_stt, des_goal) # action_dim = (1, stt_dim) -> defaults to 25-dim
                _subgoal_noise = subgoal_noise() # random sample when called
                subgoal += _subgoal_noise
                # Create a high level transition : note that the action of manager policy is subgoal
                manager_temp_transition = [[meas_stt], [aux_stt], [c_obs], [], c_obs, None, None, None, meas_stt, None, subgoal, aux_stt, None, 0, False] # TODO: no reward @ init 
                # update running mean-std normalizer
                update_rms(meas_stt=full_stt, c_obs=c_obs, aux_stt=aux_stt, act=action) # TODO : modifiy here..!

            if t % save_freq == 0 and train_indicator:
                rospy.loginfo('##### saves network weights ##### for step %d', t)
                saver.save(sess,'/home/irobot/catkin_ws/src/ddpg/scripts/ecsac/model/ecsac.ckpt', global_step=t)
                saver.save(sess, os.path.join(wandb.run.dir, '/model/ecsac.ckpt'), global_step=t)  


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=300)
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='sac')
    args = parser.parse_args()

    from utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    
    ecsac(train_indicator=1, logger_kwargs=logger_kwargs) # 1 Train / 0 Test (@real)