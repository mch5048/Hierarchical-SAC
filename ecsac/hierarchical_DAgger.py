#!/usr/bin/env python


import random
import numpy as np
import cv2
import os
import pickle
from collections import deque
import rospy
from robotReacher_v0 import robotEnv

from std_msgs.msg import *
from std_srvs.srv import *
import math
from math import degrees as deg
from math import radians as rad

from random import *
import intera_interface

from intera_core_msgs.msg import JointCommand

from tf_conversions import posemath
from tf.msg import tfMessage
from tf.transformations import quaternion_from_euler
from collections import deque

import PyKDL
from intera_interface import Limb
from intera_interface import Gripper
from running_mean_std import RunningMeanStd

from intera_interface import CHECK_VERSION
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

from std_msgs.msg import Header
from intera_motion_msgs.msg import (
    Trajectory,
    TrajectoryOptions,
    Waypoint
)

import time

# save running mean std for demo transition
from SetupSummary import SummaryManager_HIRO as SummaryManager
rms_path = '/home/irobot/catkin_ws/src/ddpg/scripts/ecsac_aux/'
demo_path = '/home/irobot/catkin_ws/src/ddpg/scripts/ecsac_aux/'



class DemoReplayBuffer(object):
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

    def return_buffer(self):
        """ Returns the whole numpy arrays containing demo transitions.
        """
        return {'data':[self.obs_buf, self.obs1_buf, self.g_buf, self.g1_buf, self.stt_buf, self.stt1_buf,
                self.act_buf, self.rews_buf, self.done_buf, self.aux_buf, self.aux1_buf]
                'size':self.size}

class DemoManagerReplayBuffer(DemoReplayBuffer):
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

        def return_buffer(self):
        """ Returns the whole numpy arrays containing demo transitions, for manager transitions
        """
        return {'data':[self.obs_buf, self.obs1_buf, self.g_buf, self.g1_buf, self.stt_buf, self.stt1_buf,
                self.act_buf, self.rews_buf, self.done_buf, self.aux_buf, self.aux1_buf],
                'seq_data':[self.stt_seq_buf, self.obs_seq_buf, self.act_seq_buf],
                'size':self.size}


def normalize(x, stats):
    if stats is None:
        return x
    return (x - stats.mean) / stats.std


def normalize_action(action_arr):
    lb_array = ACTION_LOW_BOUND*np.ones(action_arr.shape)
    hb_array = ACTION_HIGH_BOUND*np.ones(action_arr.shape)
    _norm_action = lb_array + (action_arr+1.0*np.ones(action_arr.shape))*0.5*(hb_array - lb_array)
    _norm_action = np.clip(_norm_action, lb_array, hb_array)
    _norm_action = _norm_action.reshape(action_arr.shape)
    return _norm_action


def randomize_world():
    rospy.wait_for_service('/dynamic_world_service')
    dynamic_world_service_call = rospy.ServiceProxy('/dynamic_world_service', Empty)
    change_env_request = EmptyRequest()
    dynamic_world_service_call(change_env_request)


if __name__ == '__main__':

    USE_CARTESIAN = True
    USE_GRIPPER = True

    # define observation dimensions
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

    limb = Limb()
    gripper = Gripper()
    os.chdir('/home/irobot/catkin_ws/src/ddpg/ecsac')

    # demo quantity related
    total_epi = 50
    max_ep_len = 1000
    total_steps = total_epi * max_ep_len
    buffer_size = 5e4 # 50000 steps : is it enough?
    manager_propose_freq = 10

    isDemo = True
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

    env = robotEnv(max_steps=max_ep_len, control_mode='velocity', isPOMDP=True, isGripper=USE_GRIPPER, isCartesian=USE_CARTESIAN, train_indicator=IS_TRAIN)
    controller_buffer = ReplayBuffer(obs_dim=obs_dim, stt_dim=stt_dim, act_dim=act_dim, aux_dim=aux_dim, size=buffer_size)
    manager_buffer = ManagerReplayBuffer(obs_dim=obs_dim, stt_dim=stt_dim, act_dim=act_dim, aux_dim=aux_dim, size=buffer_size, seq_len=manager_propose_freq)
        
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
        """
        return np.zeros(sub_goal_dim)


    def demo_subgoal_transition():
        """ replace the temp subgoals in temp manager buffer with expert subgoals for the rollout batches.
        TODO: implement this function!
        """
        raise NotImplementedError


    def get_action():
        """ return the action inference from the external controller.
        """
        return env._get_demo_aciton()


    while not rospy.is_shutdown() and t <int(total_steps):
        if done or ep_len== max_ep_len: # if an episode is finished (no matter done==True)
            if t != 0:
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
                
                manager_temp_transition = demo_subgoal_transition(manager_temp_transition)
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
            # args for get_subgoal : obs, next_obs, sub_goal
            subgoal = get_demo_temp_subgoal(full_stt, next_full_stt, subgoal) # action_dim = (1, stt_dim) -> defaults to 25-dim
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
                update_rms(full_stt=full_stt, c_obs=c_obs, aux=aux, act=action) # do it.

            if t % save_freq == 0 and train_indicator:
                rospy.loginfo('##### saves network weights ##### for step %d', t)
                saver.save(sess,os.getcwd()+'/src/ddpg/scripts/ecsac/model/ecsac.ckpt', global_step=t)
                saver.save(sess, os.path.join(wandb.run.dir, '/model/ecsac.ckpt'), global_step=t)  














    for e in range(episode_count):

        data_aggr_list = []
        rospy.loginfo("Now It's EPISODE{0}".format(e))
        observation = env.reset_teaching() # done-> False
        o_t = np.array(observation[0])
        s_t = [np.array(observation[1]), np.array(observation[2]), np.array(observation[3]), np.array(observation[4])]

        if e == 0:
            o_t_rms = RunningMeanStd(shape=(1,) + o_t.shape)
            s_t0_rms = RunningMeanStd(shape=(1,) + s_t[0].shape) # q
            s_t1_rms = RunningMeanStd(shape=(1,) + s_t[1].shape) # q_dot
            s_t2_rms = RunningMeanStd(shape=(1,) + s_t[2].shape) # q_ddot
            s_t3_rms = RunningMeanStd(shape=(1,) + s_t[3].shape) # obj_pos

        s_t[0] = s_t[0].reshape(1,s_t[0].shape[0])
        s_t[1] = s_t[1].reshape(1,s_t[1].shape[0])
        s_t[2] = s_t[2].reshape(1,s_t[2].shape[0])
        s_t[3] = s_t[3].reshape(1,s_t[3].shape[0])
        o_t = np.reshape(o_t,(-1,100,100,3))

        s_t = np.concatenate(s_t,axis=-1)
        s_t = np.squeeze(s_t)

        o_t = normalize(o_t, o_t_rms)
        s_t[:7] = normalize(s_t[:7], s_t0_rms)
        s_t[14:21] = normalize(s_t[14:21], s_t2_rms)
        s_t[21:] = normalize(s_t[21:], s_t3_rms)

        step = 0
        done = False
        for s in range(max_steps):

            randomize_world()
            start_time = time.time()
            rospy.sleep(0.008)
            a_t = np.array(env.getAction_Dagger())

            observation, r_t, done = env.step_teaching(step)

            o_t_1 = np.array(observation[0])
            s_t_1 = [np.array(observation[1]), np.array(observation[2]), np.array(observation[3]),  np.array(observation[4])]
            
            s_t_1[0] = s_t_1[0].reshape(1,s_t_1[0].shape[0])
            s_t_1[1] = s_t_1[1].reshape(1,s_t_1[1].shape[0])
            s_t_1[2] = s_t_1[2].reshape(1,s_t_1[2].shape[0])
            s_t_1[3] = s_t_1[3].reshape(1,s_t_1[3].shape[0])

            o_t_1 = np.reshape(o_t_1,(-1,100,100,3))
            s_t_1 = np.concatenate(s_t_1,axis=-1)
            s_t_1 = np.squeeze(s_t_1)

            _rms_o_t_1 = o_t_1[:]
            _rms_s_t_1 = s_t_1[:]

            o_t_1 = normalize(o_t_1, o_t_rms) # DO NOT NORMALIZE VISUAL OBSERVATION
            s_t_1[:7] = normalize(s_t_1[:7], s_t0_rms)
            s_t_1[7:14] = normalize(s_t_1[7:14], s_t1_rms)
            s_t_1[14:21] = normalize(s_t_1[14:21], s_t2_rms)
            s_t_1[21:] = normalize(s_t_1[21:], s_t3_rms)

            # must include isDemo for DDPGfD
            data_aggr_list.append([o_t, s_t, a_t, r_t, o_t_1, s_t_1, done, isDemo])     
            elapsed_time = time.time() - start_time

            total_time +=elapsed_time
            o_t_rms.update(_rms_o_t_1)
            s_t0_rms.update(_rms_s_t_1[:7])
            s_t1_rms.update(_rms_s_t_1[7:14])
            s_t2_rms.update(_rms_s_t_1[14:21])
            s_t3_rms.update(_rms_s_t_1[21:])


            if np.mod(s, 10) == 0:
                    print("Episode", e, "Step", step, "Action", a_t ,"10step-Time", total_time)
                    total_time = 0
            if done:
                print 'Episode done'
                break
            o_t = np.copy(o_t_1)
            s_t = s_t_1[:]

            print 
            step += 1            
        print ('End of episode'+str(e))   

        print ('extends trajectories to total list')
        data_aggr_total_list.extend(data_aggr_list)


    print ('Now saves the collected tra jectory in pickle format')
    os.chdir('/home/irobot/catkin_ws/src/ddpg/scripts')
    with open ('traj_dagger.bin', 'wb') as f:             
        pickle.dump(data_aggr_total_list, f)


