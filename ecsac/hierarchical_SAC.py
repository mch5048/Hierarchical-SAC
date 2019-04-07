#!/usr/bin/env python
import numpy as np
import tensorflow as tf
# import gym
import time as timer
import time
import hrl_core
from hrl_core import get_vars
from utils.logx import EpochLogger
from utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from Sawyer_DynaGoalEnv_SAC_v0 import robotEnv # REFACTOR TO REMOVE GOALS
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
from state_action_space import *

from wandb.tensorflow import wandb

wandb.init(project="hierarchical-sac", tensorboard=True)

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
DEMO_DIR = '/home/irobot/catkin_ws/src/ddpg/scripts/ecsac_exp'
DEMO_DATA = 'traj_dagger.bin'

# BC_MODEL_LOADDIR = '/home/irobot/catkin_ws/src/ddpg/scripts/sac_exp/bc_pretrain/weights/'

class ReplayBuffer(object):
    """
    A simple FIFO experience replay buffer for ECSAC + HIRO agents.
    For HIRO, we need seperate replay buffers for both manager and controller policies.
    low-level controller does not require state-action sequence
    high-level controller requires extra buffer for state-action 
    """

    def __init__(self, obs_dim, stt_dim, act_dim, aux_dim, size, manager=False):
        # 9 types of data in buffer
        self.obs_buf = np.zeros(shape=(size,)+ obs_dim, dtype=np.float32) # o_t, (-1,100,100,3)
        self.obs1_buf = np.zeros(shape=(size,)+ obs_dim, dtype=np.float32) # o_t+1, (-1,100,100,3)
        self.g_buf = np.zeros(shape=(size , stt_dim), dtype=np.float32) # g_t, (-1,21)
        self.g1_buf = np.zeros(shape=(size , stt_dim), dtype=np.float32) # g_t+1, (-1,21)
        self.stt_buf = np.zeros(shape=(size , stt_dim), dtype=np.float32) # s_t (-1,21), joint pos, vel, eff
        self.stt1_buf = np.zeros(shape=(size , stt_dim), dtype=np.float32) # s_t+1 for (-1,21), consists of pos, vel, eff       self.act_buf = np.zeros(shape=(size, act_dim), dtype=np.float32) # A+t 3 dim for action
        if not manager:
            self.act_buf = np.zeros(shape=(size , act_dim), dtype=np.float32) # a_t (-1,8)
        else:
            self.act_buf = np.zeros(shape=(size , stt_dim), dtype=np.float32)
        self.rews_buf = np.zeros(shape=(size,), dtype=np.float32)
        self.done_buf = np.zeros(shape=(size,), dtype=np.float32)
        self.aux_buf = np.zeros(shape=(size , aux_dim), dtype=np.float32)
        self.aux1_buf = np.zeros( shape=(size , aux_dim), dtype=np.float32)
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

class ManagerReplayBuffer(ReplayBuffer):
    """
    A simple FIFO experience replay buffer for ECSAC + HIRO agents.
    For HIRO, we need seperate replay buffers for both manager and controller policies.
    low-level controller does not require state-action sequence
    high-level controller requires extra buffer for state-action 
    """

    def __init__(self, obs_dim, stt_dim, act_dim, aux_dim,  size, seq_len):
        """full-state/ color_observation sequence lists' shape[1] is +1 longer than that of 
        action seqence -> they are stored as s_t:t+c/o_t:t+c, while action is stored as a_t:t+c
        """

        super(ManagerReplayBuffer, self).__init__(obs_dim, stt_dim, act_dim, aux_dim, size, manager=True)

        self.stt_seq_buf = np.zeros(shape=(size, seq_len+1, stt_dim), dtype=np.float32) # s_t (-1, 10+1, 21), joint pos, vel, eff
        self.obs_seq_buf = np.zeros(shape=(size, seq_len+1,)+ obs_dim, dtype=np.float32) # o_t, (-1, 10+1, 100, 100, 3)
        self.act_seq_buf = np.zeros(shape=(size, seq_len, act_dim), dtype=np.float32) # a_t (-1,10, 8)

    def store(self, stt_seq, obs_seq, act_seq, *args, **kwargs):
        """store step transition in the buffer
        """
        super(ManagerReplayBuffer, self).store(manager=True, *args, **kwargs)
        
        self.stt_seq_buf[self.ptr] = np.array(stt_seq)
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
                    st_seq=self.stt_seq_buf[idxs],
                    ot_seq=self.obs_seq_buf[idxs],
                    at_seq=self.act_seq_buf[idxs])

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


def update_summary(summary_writer = None, pi_summray_str=None, v_summray_str=None, global_step=None):
        pi_summary_str = pi_summray_str
        v_summary_str = v_summray_str
        summary_writer.add_summary(pi_summary_str, global_step)
        summary_writer.add_summary(v_summary_str, global_step)


def randomize_world():

    # We first wait for the service for RandomEnvironment change to be ready
    # rospy.loginfo("Waiting for service /dynamic_world_service to be ready...")
    rospy.wait_for_service('/dynamic_world_service')
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
    """
    Implementation of EC-SAC for Sawyer's reach & grasp
    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in placeholder symbols 
            for state, ``x_ph``, and action, ``a_ph``, and returns the main 
            outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``mu``       (batch, act_dim)  | Computes mean actions from policy
                                           | given states.
            ``pi``       (batch, act_dim)  | Samples actions from policy given 
                                           | states.
            ``logp_pi``  (batch,)          | Gives log probability, according to
                                           | the policy, of the action sampled by
                                           | ``pi``. Critical: must be differentiable
                                           | with respect to policy parameters all
                                           | the way through action sampling.
            ``q1``       (batch,)          | Gives one estimate of Q* for 
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q2``       (batch,)          | Gives another estimate of Q* for 
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q1_pi``    (batch,)          | Gives the composition of ``q1`` and 
                                           | ``pi`` for states in ``x_ph``: 
                                           | q1(x, pi(x)).
            ``q2_pi``    (batch,)          | Gives the composition of ``q2`` and 
                                           | ``pi`` for states in ``x_ph``: 
                                           | q2(x, pi(x)).
            ``v``        (batch,)          | Gives the value estimate for states
                                           | in ``x_ph``. 
            ===========  ================  ======================================
    """
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
    man_replay_size=int(5e4) # Memory leakage?
    delayed_update_freq = 4
    #controller sac
    ctrl_replay_size=int(5e4) # Memory leakage?
    target_ent=0.01 # learnable entropy
    reward_scale_lo = 1.0

    # coefs for nn ouput regularizers
    reg_param = {'lam_mean':1e-3, 'lam_std':1e-3}
    lam_preact = 0.0
    # wandb
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
    DATA_LOAD_STEP = 20000

    IS_TRAIN = train_indicator
    USE_CARTESIAN = True
    USE_GRIPPER = True

    exp_name = 'hiro-ecsac-td3'

    logger = EpochLogger(output_dir=log_path,exp_name=exp_name, seed=seed)
    logger.save_config(locals())

    tf.set_random_seed(seed)
    np.random.seed(seed)

    # arguments for RobotEnv : isdagger, isPOMDP, isGripper, isReal
    env = robotEnv(isPOMDP=True, isGripper=USE_GRIPPER, isCartesian=USE_CARTESIAN, train_indicator=IS_TRAIN)

    rospy.loginfo("Evironment has been created")
    # instantiate actor-critic nets for controller
    controller_actor_critic = hrl_core.cnn_controller_actor_critic
    rospy.loginfo("Creates actor-critic nets for low-level contorller")
    # instantiate actor-critic nets for manager
    manager_actor_critic = hrl_core.mlp_manager_actor_critic
    rospy.loginfo("Creates actor-critic nets for high-level manager")
    
    # for low_level controller
    obs_dim = (100, 100, 3) # for actor in POMDP
    stt_dim = 21# full_state of the robot (joint positions, velocities, and efforts) + ee position
    act_dim = 8 # 7 joint vels and gripper position 
    aux_dim = 3 # target object's position
    # define joint vel_limits
    action_space = (-1.0, 1.0)
    ee_dim = 7 # 4 quaternion
    grip_dim = 1 # 1 dimension for the gripper position

    # for high_level controller
    des_goal_dim = 21 # (joint positions, velocities, and efforts) + ee position
    sub_goal_dim = 21 # (joint positions, velocities, and efforts) + ee position

    if USE_CARTESIAN: # append 7-dim
        stt_dim += ee_dim
        des_goal_dim += ee_dim
        sub_goal_dim += ee_dim

    if USE_GRIPPER: # append 1-dim
        stt_dim += grip_dim
        des_goal_dim += grip_dim
        sub_goal_dim += grip_dim
   
    # Inputs to computation graph -> placeholders for next goal state is necessary.
    obs_ph = tf.placeholder(dtype=tf.float32, shape=(None,)+obs_dim) # o_t : low_level controller
    obs1_ph = tf.placeholder(dtype=tf.float32, shape=(None,)+obs_dim) # o_t+1 : low_level controller
    sg_ph = tf.placeholder(dtype=tf.float32, shape=(None, stt_dim)) # g_t_tilde : low_level controller
    sg1_ph = tf.placeholder(dtype=tf.float32, shape=(None, stt_dim)) # g_t+1_tilde : low_level controller
    stt_ph = tf.placeholder(dtype=tf.float32, shape=(None, stt_dim)) # s_t : low_level critic, high_level actor-critic
    stt1_ph = tf.placeholder(dtype=tf.float32, shape=(None, stt_dim)) # s_t+1 : low_level critic, high_level actor-critic
    dg_ph = tf.placeholder(dtype=tf.float32, shape=(None, stt_dim)) # dg_t : high_level_actor-critic -> task goal
    dg1_ph = tf.placeholder(dtype=tf.float32, shape=(None, stt_dim)) # dg_t+1 : high_level_actor-critic -> task goal
    act_ph = tf.placeholder(dtype=tf.float32, shape=(None, act_dim)) # a_t : for low_level a-c
    aux_ph = tf.placeholder(dtype=tf.float32, shape=(None, aux_dim)) # only fed into the critic
    aux1_ph = tf.placeholder(dtype=tf.float32, shape=(None, aux_dim)) # only fed into the critic
    rew_ph_lo = tf.placeholder(dtype=tf.float32, shape=(None)) # r_t(s,g,a,s') = -1*l2_norm(s+g-s') : given by high-level policy
    rew_ph_hi = tf.placeholder(dtype=tf.float32, shape=(None)) # R_t = sparse (1 if suc else 0): given by the env.
    dn_ph = tf.placeholder(dtype=tf.float32, shape=(None)) # if episode is done : given by env.
    rospy.loginfo("placeholder have been created")

    # her_sampler = Her_sampler(reward_func=env.compute_reward_from_goal)

    # training ops definition
    # define operations for training high-level policy : TD3
    with tf.variable_scope('manager'):
        with tf.variable_scope('main'):                                                     
            mu_hi, q1_hi, q2_hi, q1_pi_hi, pi_reg_hi = manager_actor_critic(stt_ph, dg_ph, sg_ph, aux_ph, action_space=None)
        # target_policy network
        with tf.variable_scope('target'):
            mu_hi_targ, _, _, _, _ = manager_actor_critic(stt1_ph, dg1_ph, sg_ph, aux_ph, action_space=None)

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
            _, q1_targ_hi, q2_targ_hi, _, _ = manager_actor_critic(stt1_ph, dg1_ph, act_hi_targ, aux_ph, action_space=None)

    # define operations for training low-level policy : SAC
    with tf.variable_scope('controller'):
        with tf.variable_scope('main'):
            mu_lo, pi_lo, logp_pi_lo, q1_lo, q2_lo, q1_pi_lo, q2_pi_lo, v_lo, pi_g, reg_losses = controller_actor_critic(stt_ph, obs_ph, sg_ph, act_ph, aux_ph, action_space=None)

        # Target value network
        with tf.variable_scope('target'):
            # _, _, _, _, _, _, _, v_targ_lo, _, _  = controller_actor_critic(stt1_ph, obs1_ph, sg1_ph, act_ph, aux1_ph, action_space=None)
            _, _, _, _, _, _, _, _, _, _  = controller_actor_critic(stt1_ph, obs1_ph, sg1_ph, act_ph, aux1_ph, action_space=None)

    # Experience buffer for seperate controller and manager
    controller_buffer = ReplayBuffer(obs_dim=obs_dim, stt_dim=stt_dim, act_dim=act_dim, aux_dim=aux_dim, size=ctrl_replay_size)
    manager_buffer = ManagerReplayBuffer(obs_dim=obs_dim, stt_dim=stt_dim, act_dim=act_dim, aux_dim=aux_dim, size=man_replay_size, seq_len=manager_propose_freq)
    
    # high-level policy training based on TD3

    # Bellman backup for Q functions, using clipped double-Q targets
    # Min Q_targ: for manager -> TD3 learning
    with tf.variable_scope('manager'):
        min_q_targ_hi = tf.minimum(q1_targ_hi, q2_targ_hi) # tensor
        backup_hi = tf.stop_gradient(rew_ph_hi + gamma*(1-dn_ph)*min_q_targ_hi)
        # Losses for TD3
        pi_loss_hi = -tf.reduce_mean(q1_pi_hi)
        pi_loss_hi += pi_reg_hi*reg_param['lam_mean'] # regularization loss for the actor
        q1_loss_hi = tf.reduce_mean((q1_hi-backup_hi)**2)
        q2_loss_hi = tf.reduce_mean((q2_hi-backup_hi)**2)
        q_loss_hi = q1_loss_hi + q2_loss_hi
        l2_loss_hi = tf.losses.get_regularization_loss()
        q_loss_hi += l2_loss_hi
    # low-level policy training based on SAC

    with tf.variable_scope('controller'):
        # Min Double-Q:
        min_q_pi_lo = tf.minimum(q1_pi_lo, q2_pi_lo)
        # Targets for Q and V regression
        log_alpha_lo = tf.get_variable(name='log_alpha', initializer=-1.0, dtype=np.float32)
        alpha_lo = tf.exp(log_alpha_lo) 

        # new
        value_targ_lo = min_q_pi_lo + - alpha_lo * logp_pi_lo # implicit valfue ftn thru soft q-ftn
        q_backup_lo = tf.stop_gradient(rew_ph_lo + gamma*(1-dn_ph)*value_targ_lo) 

        # Soft actor-critic losses
        # pi_loss_lo = tf.reduce_mean(alpha_lo * logp_pi_lo - q1_pi_lo) # grad_ascent for E[q1_pi+ alpha*H] > maximize return && maximize entropy
        pi_loss_lo = tf.reduce_mean(alpha_lo * logp_pi_lo - q1_pi_lo) # grad_ascent for E[q1_pi+ alpha*H] > maximize return && maximize entropy
        pi_loss_lo += reg_losses['preact_reg']*reg_param['lam_mean']  # regularization losses for the actor
        pi_loss_lo += reg_losses['std_reg']*reg_param['lam_std']
        # q1_loss_lo = 0.5 * tf.reduce_mean((q_backup_lo - q1_lo)**2)
        # q2_loss_lo = 0.5 * tf.reduce_mean((q_backup_lo - q2_lo)**2)
        # new
        q1_loss_lo = tf.losses.mean_squared_error(labels=q_backup_lo, predictions=q1_lo, weights=0.5)
        q2_loss_lo = tf.losses.mean_squared_error(labels=q_backup_lo, predictions=q2_lo, weights=0.5)
        q1_l2_loss_lo = tf.losses.get_regularization_loss()
        q2_l2_loss_lo = tf.losses.get_regularization_loss()
        q1_loss_lo += q1_l2_loss_lo
        q2_loss_lo += q2_l2_loss_lo
        alpha_loss_lo = tf.reduce_mean(-log_alpha_lo*tf.stop_gradient(logp_pi_lo+target_ent)) # why tf.stop_gradient here?
        # all losses make sense to me!
    
    # Policy train op
    # (has to be separate from value train op, because q1_pi appears in pi_loss)
    with tf.variable_scope('manager'): # TD3 training ops
    # Separate train ops for pi, q
        pi_hi_optimizer = tf.train.AdamOptimizer(learning_rate=pi_lr, name='pi_hi_optimizer')
        q_hi_optimizer = tf.train.AdamOptimizer(learning_rate=vf_lr, name='q_hi_optimizer')
        rospy.loginfo("manager optimizers have been created")
        train_pi_hi_op = tf.contrib.layers.optimize_loss(
            pi_loss_hi, global_step=None, learning_rate=pi_lr, optimizer=pi_hi_optimizer, variables=get_vars('manager/main/pi'),
            increment_global_step=False, summaries=("loss", "gradients", "gradient_norm", "global_gradient_norm"))

    with tf.control_dependencies([train_pi_hi_op]): # expreimental application on high-level policy's learning in TD3.
        train_q_hi_op = tf.contrib.layers.optimize_loss(
            q_loss_hi, global_step=None, learning_rate=pi_lr, optimizer=q_hi_optimizer, variables=get_vars('manager/main/q'),
            increment_global_step=False, summaries=("loss", "gradients", "gradient_norm", "global_gradient_norm"))

    with tf.control_dependencies([train_q_hi_op]): # train_qfs 
        target_update_hi = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                                for v_main, v_targ in zip(get_vars('manager/main'), get_vars('manager/target'))])
        # Polyak averaging for target variables
        step_hi_q_ops = [tf.reduce_mean(q1_hi), tf.reduce_mean(q2_hi), q1_loss_hi, q2_loss_hi, q_loss_hi, 
                        train_q_hi_op]
        step_hi_pi_ops = [pi_loss_hi, train_pi_hi_op, target_update_hi]
        step_hi_ops = {'q_ops': step_hi_q_ops, 'pi_ops':step_hi_pi_ops}
        # step_hi_ops += monitor_hi_ops
        rospy.loginfo("conputational ops for manager are instantiated")

    with tf.variable_scope('controller'): # SAC training ops
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
            increment_global_step=False, summaries=("loss", "gradients", "gradient_norm", "global_gradient_norm"))

        with tf.control_dependencies([train_pi_lo_op]): # training pi won't affect learning vf
            train_q1_lo_op = tf.contrib.layers.optimize_loss(
            q1_loss_lo, global_step=None, learning_rate=vf_lr, optimizer=q1_lo_optimizer, variables=get_vars('controller/main/q1'),
            increment_global_step=False, summaries=("loss", "gradients", "gradient_norm", "global_gradient_norm"))
            train_q2_lo_op = tf.contrib.layers.optimize_loss(
            q2_loss_lo, global_step=None, learning_rate=vf_lr, optimizer=q2_lo_optimizer, variables=get_vars('controller/main/q2'),
            increment_global_step=False, summaries=("loss", "gradients", "gradient_norm", "global_gradient_norm"))
            train_q_lo_op = tf.group(train_q1_lo_op, train_q2_lo_op)
        # Polyak averaging for target variables
        # (control flow because sess.run otherwise evaluates in nondeterministic order)
        with tf.control_dependencies([train_q_lo_op]): # train_qfs 
            target_update_lo = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                                    for v_main, v_targ in zip(get_vars('controller/main'), get_vars('controller/target'))])

        # All ops to call during one training step
        step_lo_ops = [pi_loss_lo, q1_loss_lo, q2_loss_lo, tf.reduce_mean(q1_lo), tf.reduce_mean(q2_lo), logp_pi_lo, 
        train_pi_lo_op, train_q_lo_op, target_update_lo]
        alpha_lo_op = [train_alpha_lo_op, alpha_lo] # monitor alpha_lo
        step_lo_ops +=alpha_lo_op
        rospy.loginfo("conputational ops for controller are instantiated")

    # Initializing targets to match main variables
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
            # low = np.tile(np.array(low), [batch_size,1])
            # high = np.tile(np.array(high), [batch_size,1])
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
        return 0.02*gripper + 0.022


    def get_action(obs, sub_goal, deterministic=False): # True for test
        """ infer primitive action from low-level policy
            TODO : manage when the obs and sub_goal are batch of actions
            7 joint velocities 
        """
        act_op = mu_lo if deterministic else pi_lo # returned from actor_critic function
        sub_goal, obs = normalize_observation(full_stt=sub_goal.reshape((-1,sub_goal_dim)), c_obs=obs.reshape((-1,)+obs_dim))
        # TODO: resolve shape mismatching issue
        return np.squeeze(sess.run(act_op, feed_dict={obs_ph: obs,
                                           sg_ph: sub_goal}))


    def get_subgoal(state, des_goal):
        """ infer subgoal from high-level policy for every freq.
        The output of the higher-level actor net is scaled to an approximated range of high-level acitons.
        7 joint torques : , 7 joint velocities: , 7 joint_positions: , 3-ee positions: , 4-ee quaternions: , 1 gripper_position:
        """
        state = normalize_observation(full_stt=state.reshape((-1, stt_dim)))
        des_goal = normalize_observation(full_stt=des_goal.reshape((-1,des_goal_dim)))
        return np.squeeze(sess.run(mu_hi, feed_dict={stt_ph: state,    
                                          dg_ph: des_goal}))


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
        for itr in range(ep_len):
            batch = _ctrl_buffer.sample_batch(batch_size)
            ctrl_feed_dict = {obs_ph: normalize_observation(c_obs=batch['ot']),
                         obs1_ph: normalize_observation(c_obs=batch['ot1']),
                         stt_ph: normalize_observation(full_stt=batch['st']),
                         stt1_ph: normalize_observation(full_stt=batch['st1']),
                         act_ph: normalize_observation(act=batch['acts']),
                         sg_ph : normalize_observation(full_stt=batch['g']),
                         sg1_ph : normalize_observation(full_stt=batch['g1']),
                         aux_ph : normalize_observation(aux=batch['aux']),
                         aux1_ph : normalize_observation(aux=batch['aux1']),
                         rew_ph_lo: (batch['rews']),
                         dn_ph: batch['done'],
                        }
            # low_outs = sess.run(controller_ops + monitor_lo_ops, ctrl_feed_dict)
            low_outs = sess.run(controller_ops, ctrl_feed_dict)
            # logging
            wandb.log({'policy_loss_lo': low_outs[0], 'q1_loss_lo': low_outs[1],
                        'q2_loss_lo': low_outs[2], 'q1_lo': low_outs[3], 
                        'q2_lo': low_outs[4], 'alpha': low_outs[10], 'step': step})

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
        last_states = [s[-1] for s in _s_seq] # s_t+c-1
        first_obs = [o[0] for o in _o_seq] # o_t
        last_obs = [o[-1] for o in _o_seq] # o_t+c-1
        diff_goal = (np.array(last_states)-np.array(first_states))[:,np.newaxis,:]
        original_goal = np.array(subgoals)[:, np.newaxis, :] 
        random_goals = np.random.normal(loc=diff_goal, size=(batch_size, _candidate_goals, original_goal.shape[-1])) # gaussian centered @ s_t+c - s_t

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
        flat_batch_size = seq_len*batch_size
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
        batch = _man_buffer.sample_batch(batch_size)
        # subgoal here is the action of manager network, action for computing log-likelihood
        corr_subgoals = off_policy_correction(subgoals= batch['acts'],s_seq= batch['st_seq'],
        # shape of corr_subgoals (batch_size, goal_dim)
        o_seq= batch['ot_seq'], a_seq= batch['at_seq'])
        q_ops = train_ops['q_ops'] # {'q_ops': step_hi_q_ops, 'pi_ops':step_hi_pi_ops}
        # action of the manager is subgoal...
        man_feed_dict = {stt_ph: normalize_observation(full_stt=batch['st']),
                         stt1_ph: normalize_observation(full_stt=batch['st1']),
                         sg_ph: normalize_observation(full_stt=batch['acts']),
                         dg_ph : normalize_observation(full_stt=batch['g']),
                         dg1_ph : normalize_observation(full_stt=batch['g1']),
                         aux_ph : normalize_observation(aux=batch['aux']),
                         aux1_ph : normalize_observation(aux=batch['aux1']),
                         rew_ph_hi: batch['rews'],
                         dn_ph: batch['done'],
                        }
                       
        q_ops = train_ops['q_ops'] # [q1_hi, q2_hi, q1_loss_hi, q2_loss_hi, q_loss_hi, train_q_hi_op]
        pi_ops = train_ops['pi_ops'] # [pi_loss_hi, train_pi_hi_op, target_update_hi]
        q_hi_outs = sess.run(q_ops, man_feed_dict)

        if step % delayed_update_freq ==0: # delayed update of the policy and target nets.
            pi_hi_outs = sess.run(pi_ops, man_feed_dict)
            wandb.log({'policy_loss_hi': pi_hi_outs[0]})
            rospy.loginfo('writes summary of high-level policy')

        # logging
        wandb.log({'q1_hi': q_hi_outs[0], 'q2_hi': q_hi_outs[1], 'q1_loss_hi': q_hi_outs[2],
                     'q2_loss_hi': q_hi_outs[3], 'q_loss_hi': q_hi_outs[4], 'step': step})
        rospy.loginfo('writes summary of high-level value-ftn')


    def normalize_observation(full_stt=None, c_obs=None, aux=None, act=None):
        """ normalizes observations for each step based on running mean-std.
        will only be used for each subgoal /action estimation from the manager/controller policy
        """
        if c_obs is not None and full_stt is None:
            c_obs[:] =  normalize(c_obs[:], summary_manager.s_t0_rms)
            return c_obs
        elif full_stt is not None and c_obs is None:

            full_stt[:,:7] = normalize(full_stt[:,:7], summary_manager.s_t1_rms) # joint pos
            full_stt[:,7:14] = normalize(full_stt[:,7:14], summary_manager.s_t2_rms) # joint vel
            full_stt[:,14:21] = normalize(full_stt[:,14:21], summary_manager.s_t3_rms) # joint eff
            full_stt[:,21:22] = normalize(full_stt[:,21:22], summary_manager.s_t4_rms) # gripper position
            full_stt[:,22:] = normalize(full_stt[:,22:], summary_manager.s_t5_rms) # ee pose
            return full_stt
        elif act is not None:
            act[:] = normalize(act[:], summary_manager.a_t_rms)
            return act
        elif aux is not None:
            aux[:] = normalize(aux[:], summary_manager.s_t6_rms)
            return aux
        else:
            c_obs[:] = normalize(c_obs[:], summary_manager.s_t0_rms)
            full_stt[:,:7] = normalize(full_stt[:,:7], summary_manager.s_t1_rms) # joint pos
            full_stt[:,7:14] = normalize(full_stt[:,7:14], summary_manager.s_t2_rms) # joint vel
            full_stt[:,14:21] = normalize(full_stt[:,14:21], summary_manager.s_t3_rms) # joint eff
            full_stt[:,21:22] = normalize(full_stt[:,21:22], summary_manager.s_t4_rms) # gripper position
            full_stt[:,22:] = normalize(full_stt[:,22:], summary_manager.s_t5_rms) # ee pose
            return full_stt, c_obs

    def update_rms(full_stt=None, c_obs=None, aux=None, act=None):
        """Update the mean/stddev of the running mean-std normalizers.
        Normalize full-state, color_obs, and auxiliary observation.
        Caution on the shape!
        """
        summary_manager.s_t0_rms.update(c_obs) # c_obs
        summary_manager.s_t1_rms.update(full_stt[:7]) # joint_pos
        summary_manager.s_t2_rms.update(full_stt[7:14]) # joint_vel
        summary_manager.s_t3_rms.update(full_stt[14:21]) # joint_eff
        summary_manager.s_t4_rms.update(full_stt[21:22]) # gripper_position
        summary_manager.s_t5_rms.update(full_stt[22:]) # ee_pose
        summary_manager.s_t6_rms.update(aux) # aux
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

    DEMO_USE = False # DAgger should be reimplemented 
    # os.chdir(DEMO_DIR)

    # how demo transition consists of
    # obs_t,  stt_t, act_t,   rew_t, obs_t+1, stt_t+1, done
    # TODO: implement demo acquisition into the buffer
    # TODO: implement usage of pre-trained network

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
    controller_transition = list() # temp controller transition
    if train_indicator: # train
        saver.save(sess,os.getcwd()+'/src/ddpg/scripts/ecsac/model/ecsac.ckpt', global_step=t)
    else: # test
        rospy.logwarn('Now loads the pretrained weight for test')
        load_rms()    
        new_saver = tf.train.import_meta_graph(os.getcwd()+'/src/ddpg/scripts/ecsac/model/ecsac.ckpt-{0}.meta'.format(DATA_LOAD_STEP))
        new_saver.restore(sess, os.getcwd()+'/src/ddpg/scripts/ecsac/model/ecsac.ckpt-{0}'.format(DATA_LOAD_STEP))
    # Main loop: collect experience in env and update/log each epoch
    while not rospy.is_shutdown() and t <int(total_steps):
        if done or ep_len== max_ep_len: # if an episode is finished (no matter done==True)
            if t != 0 and train_indicator:
                # train_low level controllers for the length of this episode
                train_low_level_controller(train_ops=step_lo_ops, buffer=controller_buffer, ep_len=ep_len, step=t)
        
                # train high-level manager policy in delayed manner.
                if timesteps_since_manager >= train_manager_freq:
                    timesteps_since_manager = 0
                    train_high_level_manager(train_ops=step_hi_ops, buffer=manager_buffer, ep_len=int(ep_len/train_manager_freq), step=t)

                # Process final state/obs, store manager transition i.e. state/obs @ t+c
                if len(manager_temp_transition[2]) != 1:
                    #                               0       1    2      3    4     5    6     7     8     9     10    11     12    13    
                    # buffer store arguments :    s_seq, o_seq, a_seq, obs, obs_1, dg,  dg_1, stt, stt_1, act,  aux, aux_1,  rew, done 
                    manager_temp_transition[4] = c_obs # save o_t+c
                    manager_temp_transition[8] = full_stt # save s_t+c
                    manager_temp_transition[11] = aux # save aux_t+c
                    manager_temp_transition[-1] = float(True) # done = True for manager, regardless of the episode
                # make sure every manager transition have the same length of sequence
                # TODO: debug here...
                if len(manager_temp_transition[0]) <= manager_propose_freq: # len(state_seq) = propose_freq +1 since we save s_t:t+c as a seq.
                    while len(manager_temp_transition[0]) <= manager_propose_freq:
                        manager_temp_transition[0].append(full_stt) # s_seq
                        manager_temp_transition[1].append(c_obs) # o_seq
                        manager_temp_transition[2].append(action) # a_seq
                # buffer store arguments : s_seq, o_seq, a_seq, obs, obs_1, dg, dg_1, stt, stt_1, act, aux, aux_1, rew, done 
                manager_buffer.store(*manager_temp_transition) # off policy correction is done inside the manager_buffer 
                save_rms(step=t)
            # reset the environment since an episode has been finished
            obs = env.reset() #'observation', 'desired_goal', g_des -> task specific goal!
            done = False
            reset = False
            ep_len = 0 # length of the episode
            ep_ret = 0 # episode return for the manager
            ep_low_ret = 0 # return of the intrinsic reward for low level controller 
            episode_num += 1 # for every env.reset()

            # process observations
            full_stt = np.concatenate(obs['observation']['full_state'], axis=0) # s_0
            c_obs = obs['observation']['color_obs'] #o_0
            des_goal = np.concatenate(obs['desired_goal']['full_state'], axis=0) # g_des
            aux = obs['auxiliary'] # g_des

            # infer subgoal for low-level policy
            subgoal = get_subgoal(full_stt, des_goal) # action_dim = (1, stt_dim) -> defaults to 25-dim
            _subgoal_noise = subgoal_noise() # random sample when called
            subgoal += _subgoal_noise
            timesteps_since_subgoal = 0
            # apply noise on the subgoal
            # create a temporal high-level transition : requires off-policy correction
            # buffer store arguments :    s_seq,    o_seq, a_seq, obs,  obs_1,     dg,    dg_1,      stt, stt_1, act,  aux, aux_1,  rew, done 
            manager_temp_transition = [[full_stt], [c_obs], [], c_obs, None, des_goal, des_goal, full_stt, None, subgoal, aux, None, 0, False]

        if t < low_start_timesteps:
            action = sample_action(act_dim) # a_t
        else:
            action = get_action(c_obs, subgoal, deterministic= not train_indicator) # a_t
            # TODO: make action on the gripper as categorical policy
            # action[-1] = reloc_rescale_gripper(action[-1])
        next_obs, manager_reward, done = env.step(action, time_step=ep_len) # reward R_t-> for high-level manager -> for sum(R_t:t+c-1)
        if train_indicator:
            randomize_world()
        # update episodic logs
        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state) -> if done = False for the max_timestep
        # DO NOT Make done = True when it hits timeout
        ep_len += 1
        ep_ret += manager_reward # reward in terms of achieving episodic task
        done = False if ep_len== max_ep_len else done
        if done:
            rospy.logwarn('=============== Now epsiode %d ends with done signal! ====================', episode_num)
        next_full_stt = np.concatenate(next_obs['observation']['full_state']) # s_t
        next_c_obs = next_obs['observation']['color_obs'] #o_t
        next_aux = obs['auxiliary'] # g_des

        if train_indicator:
            # append manager transition
            manager_temp_transition[-1] = float(True)
            manager_temp_transition[-2] += manager_reward # sum(R_t:t+c)
            manager_temp_transition[0].append(next_full_stt) # append s_seq
            manager_temp_transition[1].append(next_c_obs) # append o_seq
            manager_temp_transition[2].append(action)     # append a_seq        
            # compute intrinsic reward
            intrinsic_reward = env.compute_intrinsic_reward(full_stt, next_full_stt, subgoal)
        # subgoal transition
        next_subgoal = env.env_goal_transition(full_stt, next_full_stt, subgoal)

        # rospy.logwarn('=====================================================================')
        # rospy.logerr('INTRINSIC_REWARD')
        # print (intrinsic_reward)             
        # rospy.logerr('FULL_STT')
        # print (np.linalg.norm(full_stt))
        # rospy.logerr('NEXT_FULL_STT')
        # print (np.linalg.norm(full_stt))
        # rospy.logerr('SUBGOAL')
        # print (np.linalg.norm(subgoal))        
        # print (subgoal.shape)        
        # rospy.logerr('NXT_SG_NORM')
        # print (np.linalg.norm(next_subgoal))
        # print (next_subgoal.shape)
        # rospy.logwarn('=====================================================================')

        # add transition for low-level policy
        # (obs, obs1, sg, sg1, stt, stt1, act, aux, rew, done)
        if train_indicator:
            controller_buffer.store(c_obs, next_c_obs, subgoal, 
            next_subgoal, full_stt, next_full_stt, action, aux, next_aux, intrinsic_reward, done)
        # update observations and subgoal
        obs = next_obs
        subgoal = next_subgoal
        aux = next_aux

        # update logging steps
        ep_len += 1
        t +=1
        timesteps_since_manager += 1
        timesteps_since_subgoal += 1

        if train_indicator:        
            if timesteps_since_subgoal % manager_propose_freq == 0:
                # for every c-step, renew the subgoal estimation from the manager policy.
                timesteps_since_subgoal = 0
                manager_temp_transition[4] = c_obs # save o_t+c
                manager_temp_transition[8] = full_stt # save s_t+c
                manager_temp_transition[11] = aux # save aux_t+c
                manager_temp_transition[-1] = float(True) # done = True for manager, regardless of the episode 
                
                # intentional seq appending is not required here since it always satisfies c step.
                manager_buffer.store(*manager_temp_transition)
                subgoal = get_subgoal(full_stt, des_goal) # action_dim = (1, stt_dim) -> defaults to 25-dim
                _subgoal_noise = subgoal_noise() # random sample when called
                subgoal += _subgoal_noise
                # Create a high level transition : note that the action of manager policy is subgoal
                # buffer store arguments :    s_seq,    o_seq, a_seq, obs,  obs_1,     dg,    dg_1,      stt, stt_1, act,  aux, aux_1,  rew, done 
                manager_temp_transition = [[full_stt], [c_obs], [], c_obs, None, des_goal, des_goal, full_stt, None, subgoal, aux, None, 0, False]
                # update running mean-std normalizer
                # update_rms(full_stt=full_stt, c_obs=c_obs, aux=aux, act=action) # do it.

            if t % save_freq == 0 and train_indicator:
                rospy.loginfo('##### saves network weights ##### for step %d', t)
                saver.save(sess,os.getcwd()+'/src/ddpg/scripts/ecsac/model/ecsac.ckpt', global_step=t)
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