#!/usr/bin/env python
import sys
import os
import rospy
import rospkg
import numpy as np
# import super class
from Sawyer_DynaGoalEnv_SAC_v0 import robotEnv
import time
from time import sleep
from intera_core_msgs.msg import JointCommand
from geometry_msgs.msg import Pose, Point, Quaternion
# control modes
POSITION_MODE = 1
VELOCITY_MODE = 2
TORQUE_MODE = 3
POLICY_INFER_TIME = 0.01 # time took for the policy network to infer action (feed forward)

overhead_orientation = Quaternion(
                            x=-0.00142460053167,
                            y=0.999994209902,
                            z=-0.00177030764765,
                            w=0.00253311793936)
ACTION_DIM = 8 # Cartesian
OBS_DIM = (100,100,3)      # POMDP
STATE_DIM = 24        # MDP
GRIPPER_UPPER = 0.041667
GRIPPER_LOWER = 0.0

TERM_THRES = 50
SUC_THRES = 50



class demoEnv(robotEnv): 
    def __init__(self, max_steps=700, control_mode='velocity', isdagger=False, isPOMDP=False, isGripper=False, isCartesian=True, train_indicator=1):
        """RL environment class for collecting experte demos for hierarchical SAC
        """
        robotEnv.__init__(self, max_steps=max_steps, 
                        isdagger=isdagger, isPOMDP=isPOMDP, isGripper=isGripper, 
                        isCartesian=isCartesian, train_indicator=train_indicator)
        # define attributes for demo collection
        self.control_mode = control_mode
        self.control_command = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # j0~j6, gripper on/off 
        rospy.Subscriber('/robot/limb/right/joint_command', JointCommand, self.jointCommandCB)
        if rospy.has_param('vel_calc'):
            rospy.delete_param('vel_calc')

    def jointCommandCB(self, msg):
        """POSITION_MODE=1
           VELOCITY_MODE=2
           TORQUE_MODE=3
           the last element is binary (gripper on-off)
        """
        if self.control_mode=='velocity' or msg.mode==VELOCITY_MODE:
            if len(msg.velocity)==ACTION_DIM-1:
                self.control_command[:-1] = msg.velocity
        elif self.control_mode=='position' or msg.mode==POSITION_MODE:
            if len(msg.position)==ACTION_DIM-1:
                self.control_command[:-1] = msg.position
        elif self.control_mode=='torque' or msg.mode==TORQUE_MODE:
            if len(msg.effort)==ACTION_DIM-1:
                self.control_command[:-1] = msg.effort
        else:
            raise NotImplementedError


    def reset_demo(self):
        """Reset the environment for demo collection episodes.
            same functionality as "reset" method.
        """
        # Desired goal is given here.
        # Should wait until reset finishes.
        if rospy.has_param('vel_calc'):
            rospy.delete_param('vel_calc')
        self.control_command[-1] = float(0)
        if not self.isReal: # for simulated env.
            des_goal = self._reset_gazebo()
        else:
            des_goal = self._reset_real()
            
        _joint_pos, _joint_vels, _joint_effos = self.get_joints_states()
        _color_obs = self.get_color_observation()
        _targ_obj_obs = self.get_target_obj_obs()
        _gripper_pos = self.get_gripper_position()
        _ee_pose = self.get_end_effector_pose()
        obs = dict()
        obs['full_state'] = [_joint_pos, _joint_vels, _joint_effos] # default observations
        if self.isGripper:
            obs['full_state'].append([_gripper_pos])
        if self.isCartesian:
            obs['full_state'].append(_ee_pose)
        if self.isPOMDP:
            obs['color_obs'] = _color_obs
        rospy.set_param('vel_calc','true')
        return {'observation': obs,'desired_goal':des_goal, 'auxiliary':_targ_obj_obs}


    def _get_demo_action(self):
        """ Return the controll command generated by auxiliary controller (vel/torque)
        """
        return np.array(self.control_command)


    def _get_demo_subgoal(self, obs, next_obs, sub_goal):
        """ Return the subgoal from the implicit manager. Since demo is optimal trajectory,
        the value from the subgoal_transition can be returned.
        """
        return self.env_goal_transition(obs, next_obs, sub_goal)


    def grasp_object(self):
        """ Grasp the target object if reached to the target object.
        """
        self.close_gripper()
        self.control_command[-1] = float(1)


    def step_demo(self, action=None, time_step=1000):
        """Step the environment for demo collection episodes.
            :param action:
            :return: obs, reward, done
        """
        if action is not None:
            sleep(POLICY_INFER_TIME)            
        # below for step
        self.prev_tic = self.tic
        self.tic = time.time()
        self.elapsed =  time.time()-self.prev_tic
        self.done = False
        if time_step == self.max_steps:
            self.done = True
        if not self.isReal:
            curDist = self._get_dist()
            self.reward = self._compute_reward()
            if self.reward==1.0:
                self.success_count +=1
                if self.success_count == SUC_THRES:
                    print ('======================================================')
                    print ('Succeeds current Episode : SUCCEEDED')
                    print ('======================================================')
                    self.done = True
            if self._check_for_termination():
                print ('======================================================')
                print ('Terminates current Episode : OUT OF BOUNDARY')
                print ('======================================================')
                self.done = True
        #observations : joint states, color observations, e.e. pos
        _joint_pos, _joint_vels, _joint_effos = self.get_joints_states()
        _color_obs = self.get_color_observation()
        _targ_obj_obs = self.get_target_obj_obs()
        _gripper_pos = self.get_gripper_position()
        _ee_pose =  self.get_end_effector_pose()
        # obj_pos = self._get_target_obj_obs() # TODO: implement this function call.                  
        if np.mod(time_step, 10)==0:
            if not self.isReal:
                print("DISTANCE : ", curDist)
            print("PER STEP ELAPSED : ", self.elapsed)
            print("SPARSE REWARD : ", self.reward_rescale * self.reward)
            print("Current EE pos: ", self.endpt_pos)
            print("Actions: ", action)
        # achieved goal should be equivalent to the step observation dict.
        goal_obs = dict()
        obs = dict()
        obs['full_state'] = [_joint_pos, _joint_vels, _joint_effos] # default observations
        goal_obs['full_state'] = [_joint_pos, _joint_vels, _joint_effos] # default goal_observations
        if self.isGripper:
            obs['full_state'].append([_gripper_pos])
            goal_obs['full_state'].append([_gripper_pos])
        if self.isCartesian:
            obs['full_state'].append(_ee_pose)
            goal_obs['full_state'].append(_ee_pose)
        if self.isPOMDP:
            obs['color_obs'] = _color_obs
            goal_obs['color_obs'] = _color_obs
        return {'observation': obs,'achieved_goal':goal_obs, 'auxiliary':_targ_obj_obs}, self.reward_rescale*self.reward, self.done


    def reach_target_obj_vel_demo(self, target_pose):
        """ Servo the robot to reach target object with velocity controller.
        TODO: check the necessaity of making another thread.
        """
        _target_pose = target_pose
        rospy.set_param('vel_calc','true')
        self.servo_vel(_target_pose)


    def _stop_vel_controller(self):
        """ Stop the velocity controller for servoing the robot.
        """