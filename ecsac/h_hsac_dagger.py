#!/usr/bin/env python

import numpy as np
from Sawyer_PickAndPlace_DAgger import demoEnv # REFACTOR TO REMOVE GOALS
import rospy
# Leveraging demonstration is same as TD3


from std_srvs.srv import Empty, EmptyRequest
from running_mean_std import RunningMeanStd
from SetupSummary import SummaryManager_HIRO as SummaryManager
import pickle
import os
from collections import OrderedDict
from state_action_space_h_hsac import *
# intera imports 
from intera_interface import CHECK_VERSION
from intera_interface import Limb
from intera_interface import Gripper

from tf_conversions import posemath
from tf.msg import tfMessage
from tf.transformations import quaternion_from_euler
from intera_core_msgs.msg import (
    DigitalIOState,
    DigitalOutputCommand,
    IODeviceStatus
)

from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)

import time as timer
import time

from std_msgs.msg import Header
# save running mean std for demo transition
from SetupSummary import SummaryManager_HIRO as SummaryManager
rms_path = '/home/irobot/catkin_ws/src/ddpg/scripts/ecsac_aux/'
demo_path = '/home/irobot/catkin_ws/src/ddpg/scripts/ecsac_aux/'
MAN_BUF_FNAME = 'demo_manager_buffer.bin'
CON_BUF_FNAME = 'demo_controller_buffer.bin'

class DemoReplayBuffer(object):
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


    def get_episodic_subgoal(self, global_step, ep_len):
        """ Return the subgoal and fullstate batch of the episode.
        <s, s', g, g', r>
        slicing index : buf[global_step - ep_len:global_step-1] => slice out only the transitions for this episode.
        """
        return [self.stt_buf[global_step-ep_len:global_step], self.stt1_buf[global_step-ep_len:global_step], 
                self.g_buf[global_step-ep_len:global_step], self.g1_buf[global_step-ep_len:global_step],
                self.rews_buf[global_step-ep_len:global_step]]

    def calibrate_episodic_data(self, calib_batch, global_step, ep_len):
        """ calibrate the episodic data 
        <g, g', r> => tuple
        """
        self.g_buf[global_step-ep_len:global_step] = calib_batch['g_t']
        self.g1_buf[global_step-ep_len:global_step] = calib_batch['g_t+1']
        self.rews_buf[global_step-ep_len:global_step] = calib_batch['r_t']

    def return_buffer(self):
        """ Returns the whole numpy arrays containing demo transitions.
        """
        return {'data':[self.obs_buf[:self.ptr], self.obs1_buf[:self.ptr], self.g_buf[:self.ptr], 
                        self.g1_buf[:self.ptr], self.stt_buf[:self.ptr], self.stt1_buf[:self.ptr],
                        self.act_buf[:self.ptr], self.rews_buf[:self.ptr], self.done_buf[:self.ptr],
                        self.aux_buf[:self.ptr], self.aux1_buf[:self.ptr]],
                'size':self.size}


class DemoManagerReplayBuffer(DemoReplayBuffer):
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

        super(DemoManagerReplayBuffer, self).__init__(obs_dim, meas_stt_dim, act_dim, aux_stt_dim, size, manager=True)

        self.meas_stt_seq_buf = np.zeros(shape=(size, seq_len + 1, meas_stt_dim), dtype=np.float32) # s_t (-1, 10+1, 21), joint pos, vel, eff
        self.aux_stt_seq_buf = np.zeros(shape=(size, seq_len + 1, aux_stt_dim), dtype=np.float32) # s_t (-1, 10+1, 21), joint pos, vel, eff
        self.obs_seq_buf = np.zeros(shape=(size, seq_len + 1,)+ obs_dim, dtype=np.float32) # o_t, (-1, 10+1, 100, 100, 3)
        self.act_seq_buf = np.zeros(shape=(size, seq_len, act_dim), dtype=np.float32) # a_t (-1,10, 8)

    def store(self, meas_stt_seq, aux_stt_seq, obs_seq, act_seq, *args, **kwargs):
        """store step transition in the buffer
        """
        super(DemoManagerReplayBuffer, self).store(manager=True, *args, **kwargs)
        
        self.meas_stt_seq_buf[self.ptr] = np.array(meas_stt_seq)
        self.aux_stt_seq_buf[self.ptr] = np.array(aux_stt_seq)
        self.obs_seq_buf[self.ptr] = np.array(obs_seq)
        self.act_seq_buf[self.ptr] = np.array(act_seq)
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def return_buffer(self):
        """ Returns the whole numpy arrays containing demo transitions, for manager transitions
        """
        return {'data':[self.obs_buf[:self.ptr], self.obs1_buf[:self.ptr], self.g_buf[:self.ptr],
                        self.g1_buf[:self.ptr], self.stt_buf[:self.ptr], self.stt1_buf[:self.ptr],
                        self.act_buf[:self.ptr], self.rews_buf[:self.ptr], self.done_buf[:self.ptr],
                        self.aux_buf[:self.ptr], self.aux1_buf[:self.ptr]],
                'seq_data':[self.meas_stt_seq_buf[:self.ptr], self.aux_stt_seq_buf[:self.ptr], 
                            self.obs_seq_buf[:self.ptr], self.act_seq_buf[:self.ptr]],
                'size':self.size}


def normalize(x, stats):
    if stats is None:
        return x
    return (x - stats.mean) / stats.std


# def normalize_action(action_arr):
#     lb_array = ACTION_LOW_BOUND*np.ones(action_arr.shape)
#     hb_array = ACTION_HIGH_BOUND*np.ones(action_arr.shape)
#     _norm_action = lb_array + (action_arr+1.0*np.ones(action_arr.shape))*0.5*(hb_array - lb_array)
#     _norm_action = np.clip(_norm_action, lb_array, hb_array)
#     _norm_action = _norm_action.reshape(action_arr.shape)
#     return _norm_action


def randomize_world():
    """ Domain randomization for the environment's light and the color of robot link.
    """
    rospy.wait_for_service('/dynamic_world_service') # randomize the light in the gazebo world
    dynamic_world_service_call = rospy.ServiceProxy('/dynamic_world_service', Empty)
    change_env_request = EmptyRequest()
    dynamic_world_service_call(change_env_request)

    # rospy.wait_for_service('/colorize_world_service') # randomize the model colors in the gazebo world
    # colorize_world_service_call = rospy.ServiceProxy('/colorize_world_service', Empty)
    # colorize_env_request = EmptyRequest()
    # colorize_world_service_call(colorize_env_request)

if __name__ == '__main__':

    USE_CARTESIAN = True
    USE_GRIPPER = True
    IS_TRAIN = True # always true for DAgger script

    # define observation dimensions
    # for low_level controller
    obs_dim = (100, 100, 3) # for actor in POMDP
    meas_stt_dim = 21# full_state of the robot (joint positions, velocities, and efforts) + ee position
    act_dim = 7 # 7 joint vels and gripper position 
    aux_dim = 3 # target object's position
    # define joint vel_limits
    action_space = (-1.0, 1.0)
    ee_dim = 7 # 4 quaternion
    grip_dim = 1 # 1 dimension for the gripper position

    # for high_level controller
    des_goal_dim = 21 # (joint positions, velocities, and efforts) + ee position
    sub_goal_dim = 21 # (joint positions, velocities, and efforts) 

    if USE_CARTESIAN: # append 7-dim
        aux_dim += ee_dim
        des_goal_dim += ee_dim
        # sub_goal_dim += ee_dim

    if USE_GRIPPER: # append 1-dim
        meas_stt_dim += grip_dim
        des_goal_dim += grip_dim
        sub_goal_dim += grip_dim
        act_dim += grip_dim

    # init node
    # rospy.init_node('hierarchical_DAgger')

    # demo quantity related
    total_epi = 20
    max_ep_len = 500
    total_steps = total_epi * max_ep_len
    buffer_size = int(5e4) # 50000 steps : is it enough?
    manager_propose_freq = 10

    isDemo = True
    ep_ret, ep_len = 0, 0
    t = 0 # counted steps [0:total_steps - 1]
    timesteps_since_manager = 0 # to count c-step elapse for training manager
    timesteps_since_subgoal = 0 # to count c-step elapse for subgoal proposal
    episode_num = 0 # incremental episode counter
    done = True
    reset = False
    manager_temp_transition = list() # temp manager transition

    # demoEnv inherits robotEnv
    env = demoEnv(max_steps=max_ep_len, control_mode='velocity', isPOMDP=True, isGripper=USE_GRIPPER, isCartesian=USE_CARTESIAN, train_indicator=IS_TRAIN)
    controller_buffer = DemoReplayBuffer(obs_dim=obs_dim, meas_stt_dim=meas_stt_dim, act_dim=act_dim, aux_stt_dim=aux_dim, size=buffer_size)
    manager_buffer = DemoManagerReplayBuffer(obs_dim=obs_dim, meas_stt_dim=meas_stt_dim, act_dim=act_dim, aux_stt_dim=aux_dim, size=buffer_size, seq_len=manager_propose_freq)
    obs_shape_list = [(100,100,3), (7), (7), (7), (1), (7), (3)]
    summary_manager = SummaryManager(obs_shape_list=obs_shape_list) # manager for rms

    # create instances for the arm and gripper
    limb = Limb()
    gripper = Gripper()

    def update_rms(meas_stt=None, c_obs=None, aux_stt=None, act=None):
        """Update the mean/stddev of the running mean-std normalizers.
        Normalize full-state, color_obs, and auxiliary observation.
        Caution on the shape!
        """
        summary_manager.s_t0_rms.update(c_obs) # c_obs (100, 100, 3)
        summary_manager.s_t1_rms.update(meas_stt[:7]) # joint_pos (7,)
        summary_manager.s_t2_rms.update(meas_stt[7:14]) # joint_vel (7,)
        summary_manager.s_t3_rms.update(meas_stt[14:21]) # joint_eff (7,)
        summary_manager.s_t4_rms.update(meas_stt[21:]) # gripper_position (1,)
        summary_manager.s_t5_rms.update(aux_stt[3:]) # ee_pose (7,)
        summary_manager.s_t6_rms.update(aux_stt[:3]) # aux (3,)
        summary_manager.a_t_rms.update(act) # ee_pose (8,) -> qvel + grip on-off(binary)


    def load_rms():
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
        summary_manager.s_t0_rms.save_mean_std(rms_path+'mean_std0_demo.bin')
        summary_manager.s_t1_rms.save_mean_std(rms_path+'mean_std1_demo.bin')
        summary_manager.s_t2_rms.save_mean_std(rms_path+'mean_std2_demo.bin')
        summary_manager.s_t3_rms.save_mean_std(rms_path+'mean_std3_demo.bin')
        summary_manager.s_t4_rms.save_mean_std(rms_path+'mean_std4_demo.bin')
        summary_manager.s_t5_rms.save_mean_std(rms_path+'mean_std5_demo.bin')
        summary_manager.s_t6_rms.save_mean_std(rms_path+'mean_std6_demo.bin')
        summary_manager.a_t_rms.save_mean_std(rms_path+'mean_std7_demo.bin')
        

    def get_demo_temp_subgoal():
        """ return the temporary subgoal for demo transition.
        TODO : is it necessary?
        """
        return np.zeros(sub_goal_dim)


    def demo_manager_sg_transition(manager_transition):
        """ This function is called every C-step.
            replace the temp subgoals in temp manager buffer with expert subgoals for the rollout batches.
            The demo agent achieves the subgoal for every step, i.e. h = s + g - s'
                g_t == s_t+c (action of the manager)
                # g_kc = s_(k+1)c - s_kc , k= 0,1,2,... (for measured state!)
            *manager_transition :
                [[meas_stt_seq], [aux_stt_seq], [c_obs_seq], [act_seq], c_obs, c_obs_c, None, None, meas_stt, meas_stt_c, subgoal, aux_stt, aux_stt_c, 0, False]
            replace the temp. subgoal (a_t of the manager) with s_t+c-1
        """
        manager_transition[-5] = manager_transition[0][-1] - manager_transition[0][0]
        return manager_transition


    def demo_controller_sg_transition(controller_buffer, global_step, ep_len):
        """ This function is called at the end of each episode.
            *controller_buffer :
                controller_buffer.store(c_obs, next_c_obs, subgoal, 
                next_subgoal, full_stt, next_full_stt, action, aux, next_aux, intrinsic_reward, done)
            subgoal for the controller -> for every C-step sequence,
            * h = s + g - s'
                g_t == s_t+c, g' = h(s,g,s'), g'' = h(s',g',s'') ...
            *don't do subgoal_transition and intrinsic reward computation while doing rollouts.    
            controller transition (s_t||s_t+c & s_t+1||s_t+c+1) -> TD learning
            in terms of controller -> g_t:t+c-1
            index : 2 & 3
        """
        # s_t+c should be the g_t for s_t. (e.g. s_10 == g_0 -> induces new proposal)
        # TODO :  implement asserting the shape below
        sb, s1b, gb, g1b, rb = controller_buffer.get_episodic_subgoal(global_step, ep_len) # returns each batch of the transition e<s, s', g, g'>

        # ep_len - ep_len % manager_propose_freq
        # Example : if an episode is 647 length, iterate till 640 (idx 639). Then, gb[640:646] should all be the terminal state. 
        # 1. replace the subgoal proposals in 'gb'
        remainder = sb.shape[0] % manager_propose_freq
        for idx in range(0, sb.shape[0] - remainder - manager_propose_freq, manager_propose_freq): # iterate until the full proposal period is met.
            # print (idx)
            # gb[idx] = s1b[idx + manager_propose_freq - 1] - s1b[idx - 1] # s1b[idx + manager_propose_freq - 1] has the s_(idx + manager_propose_freq)
            gb[idx] = sb[idx + manager_propose_freq] - sb[idx] # g_kc = s_(k+1)c - s_kc , k= 0,1,2,...
            rb[idx] = env.compute_intrinsic_reward(sb[idx], s1b[idx + 1], gb[idx])
            for i in range(1, manager_propose_freq): #[t+1:t+c-1]
                # gb[idx + i] = env.env_goal_transition(sb[idx + i], s1b[idx + i], gb[idx])
                gb[idx + i] = env.env_goal_transition(sb[idx], s1b[idx + i], gb[idx])
                rb[idx + i] = env.compute_intrinsic_reward(sb[idx + i], s1b[idx + i + 1], gb[idx + i])
        # 2. fill ther remaining transitions with terminal state observations
        # here, gb[-1], gb[-2], ... gb[-7] = sT in example. 
        sT = s1b[-1]
        for idx in range(1,remainder + 1):
            # gb[-idx] = sT
            gb[-idx] = sT - sb[-remainder - 1 + idx]
        # 3. copy the gb into g1b, with the index offset of 1. Then the last element of g1b is sT.
        g1b[:-1] = gb[1:]
        g1b[-1] = np.ones(sT.shape)
        # recompute reward here!, and substitue the demo
        controller_buffer.calibrate_episodic_data({'g_t': gb, 'g_t+1': g1b,'r_t': rb}, global_step, ep_len)

           
    def get_action():
        """ return the action inference from the external controller.
        """
        return env._get_demo_action()
        

    # divide the loop into two phase.
    # 1. rollout (collecting normal transition data (s, a, r, s', d))
    while not rospy.is_shutdown() and t <int(total_steps):

        if done or ep_len == max_ep_len: # if an episode is finished (no matter done==True)
            if t != 0:
                # Process final state/obs, store manager transition i.e. state/obs @ t+c
                if len(manager_temp_transition[3]) != 1:
                    #                               0                1              2           3       4       5        6     7     8           9        10        11       12     13    14    
                    # buffer store arguments :  [[meas_st_seq], [aux_st_seq], [c_obs_seq], [act_seq], c_obs, c_obs_c, None, None, meas_stt, meas_stt_c, subgoal, aux_stt, aux_stt_c,  0, False]   
                    manager_temp_transition[5] = c_obs # save o_t+c ** abstracted next state!
                    manager_temp_transition[9] = meas_stt # save s_t+c ** abstracted next state!
                    manager_temp_transition[12] = aux_stt # save aux_t+c ** abstracted next state!
                    manager_temp_transition[-1] = done # done = True for manager, regardless of the episode
                # make sure every manager transition have the same length of sequence
                # TODO: debug here...
                if len(manager_temp_transition[0]) <= manager_propose_freq: # there's equal sign! len(state_seq) = propose_1freq +1 since we save s_t:t+c as a seq.
                    while len(manager_temp_transition[0]) <= manager_propose_freq:
                        manager_temp_transition[0].append(meas_stt) # s_seq
                        manager_temp_transition[1].append(aux_stt) # s_seq
                        manager_temp_transition[2].append(c_obs) # o_seq
                        manager_temp_transition[3].append(action) # a_seq0
                manager_temp_transition = demo_manager_sg_transition(manager_temp_transition)
                demo_controller_sg_transition(controller_buffer, t, ep_len) # verify if the member of the buffer class is successfully modified.
                # buffer store arguments : s_seq, o_seq, a_seq, obs, obs_1, dg, dg_1, stt, stt_1, act, aux, aux_1, rew, done 
                # print (manager_temp_transition[2])
                manager_buffer.store(*manager_temp_transition) # off policy correction is done inside the manager_buffer 
                save_rms(step=t)
            # reset the environment since an episode has been finished
            obs = env.reset_demo() #'observation', 'desired_goal', g_des -> task specific goal!
            done = False
            reset = False
            ep_len = 0 # length of the episode
            ep_ret = 0 # episode return for the manager
            ep_low_ret = 0 # return of the intrinsic reward for low level controller 
            episode_num += 1 # for every env.reset()
            # process observations
            meas_stt = np.concatenate(obs['observation']['meas_state'], axis=0) # s_0
            c_obs = obs['observation']['color_obs'] #o_0
            des_goal = np.concatenate(obs['desired_goal']['full_state'], axis=0) # g_des
            aux_stt = np.concatenate(obs['observation']['auxiliary'], axis=0) # g_des
            full_stt = np.concatenate([meas_stt, aux_stt], axis=0)

            # infer subgoal for low-level policy
            # args for get_subgoal : obs, next_obs, sub_goal
            subgoal = get_demo_temp_subgoal() # action_dim = (1, stt_dim) -> defaults to 25-dim
            timesteps_since_subgoal = 0
            # apply noise on the subgoal
            # create a temporal high-level transition : requires off-policy correction
            # buffer store arguments : meas_s_seq, aux_s_seq, o_seq, a_seq, obs, obs_c, dg,   dg_c, meas_stt, meas_stt_c, hi_act, aux, aux_c,  rew,    done 
            #                               0           1           2       3    4      5     6
            manager_temp_transition = [[meas_stt], [aux_stt], [c_obs], [], c_obs, None, None, None, meas_stt, None, subgoal, aux_stt, None,  0,   False]

        action = get_action() # a_t
        # TODO: make action on the gripper as categorical policy
        # action[-1] = reloc_rescale_gripper(action[-1])
        next_obs, manager_reward, done = env.step_demo(action, time_step=ep_len) # reward R_t-> for high-level manager -> for sum(R_t:t+c-1)
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
        next_meas_stt = np.concatenate(next_obs['observation']['meas_state'], axis=0) # s_t
        next_c_obs = next_obs['observation']['color_obs'] #o_t
        next_aux_stt = np.concatenate(next_obs['observation']['auxiliary'], axis=0) # s_aux
        next_full_stt = np.concatenate([next_meas_stt, next_aux_stt], axis=0) 

        # append manager transition
        manager_temp_transition[-1] = done # TODO verfiy
        manager_temp_transition[-2] += manager_reward # sum(R_t:t+c)
        manager_temp_transition[0].append(next_meas_stt) # append s_seq
        manager_temp_transition[1].append(next_aux_stt) # append o_seq
        manager_temp_transition[2].append(next_c_obs)     # append a_seq        
        manager_temp_transition[3].append(action)     # append a_seq        
        # compute intrinsic reward
        intrinsic_reward = env.compute_intrinsic_reward(meas_stt, next_meas_stt, subgoal)

        # subgoal transition
        next_subgoal = env.env_goal_transition(meas_stt, next_meas_stt, subgoal)
        # add transition for low-level policy
        # (obs, obs1, sg, sg1, stt, stt1, act, aux, rew, done)
        # ( obs, obs1, g, g1, stt, stt1, act, aux, aux1, rew, done, manager=False):
        controller_buffer.store(c_obs, next_c_obs, subgoal, 
        next_subgoal, meas_stt, next_meas_stt, action, aux_stt, next_aux_stt, intrinsic_reward, done) # check the sequence of replay buffer..!
        # update observations and subgoal
        obs = next_obs
        meas_stt = next_meas_stt
        aux_stt = next_aux_stt
        subgoal = next_subgoal
        full_stt = next_full_stt

        # update logging steps
        ep_len += 1
        t += 1
        timesteps_since_manager += 1
        timesteps_since_subgoal += 1

        if timesteps_since_subgoal % manager_propose_freq == 0:
            # for every c-step, renew the subgoal estimation from the manager policy.
            timesteps_since_subgoal = 0
            manager_temp_transition[5] = c_obs # save o_t+c
            manager_temp_transition[9] = meas_stt # save s_t+c
            manager_temp_transition[12] = aux_stt # save aux_t+c
            manager_temp_transition[-1] = done # done = True for manager, regardless of the episode 
            
            manager_buffer.store(*manager_temp_transition)
            subgoal = get_demo_temp_subgoal() # action_dim = (1, stt_dim) -> defaults to 25-dim
            # Create a high level transition : note that the action of manager policy is subgoal
            # buffer store  indicies :      0           1       2       3    4      5    6      7       8       9      10       11      12  13  14
            manager_temp_transition = [[meas_stt], [aux_stt], [c_obs], [], c_obs, None, None, None, meas_stt, None, subgoal, aux_stt, None, 0, False] # TODO: no reward @ init 

            # update running mean-std normalizer
            update_rms(meas_stt=meas_stt, c_obs=c_obs, aux_stt=aux_stt, act=action) # do it.

    # if all the demo episodes have ended.

    os.chdir(demo_path)
    if os.path.exists(MAN_BUF_FNAME):
        ans = input("Delete the manager buffer? '1' / '0' :")
        if ans == 1:
            print ('Deletes the manger buffer')
            os.remove(MAN_BUF_FNAME)
    if os.path.exists(CON_BUF_FNAME):
        ans = input("Delete the controller buffer? '1' / '0' :")
        if ans == 1:
            print ('Deletes the controller buffer')
            os.remove(CON_BUF_FNAME)

    print ('Now saves the manager buffer in pickle format')
    with open (MAN_BUF_FNAME, 'wb') as f:             
        pickle.dump(manager_buffer.return_buffer(), f)
    print ('Now saves the controller buffer in pickle format')
    with open (CON_BUF_FNAME, 'wb') as f2:             
        pickle.dump(controller_buffer.return_buffer(), f2)


                                                                                                                                