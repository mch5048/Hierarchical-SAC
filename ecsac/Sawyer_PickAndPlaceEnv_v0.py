#!/usr/bin/env python
import sys
import os
import rospy
import rospkg
import numpy as np
# reads open manipulator's state
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import *
from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from gazebo_msgs.msg import ContactsState
from gazebo_msgs.msg import ModelState
import math
import cv2
from cv_bridge import CvBridge, CvBridgeError
from string import Template
import time
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)

from gazebo_msgs.srv import (
    SetModelState,
    GetModelState,
    SpawnModel,
    DeleteModel,
)

from intera_core_msgs.srv import (
    SolvePositionFK,
    SolvePositionFKRequest,
    SolvePositionIK,
    SolvePositionIKRequest,
)


from controllers_connection import ControllersConnection
import modern_robotics as mr
import tf
from utils.common import rate_limited


base_dir = os.path.dirname(os.path.realpath(__file__))

# For gazebo initialization
# FK and IK related terms
home_pose_dict = { 
                    'right_j0':-0.020025390625 , 
                    'right_j1':0.8227529296875,
                    'right_j2':-2.0955126953125, 
                    'right_j3':2.172509765625,
                    'right_j4':0.7021171875,
                    'right_j5' :-1.5003603515625,
                    'right_j6' : -2.204990234375}

starting_joint_angles = {'right_j0': -0.041662954890248294,
                            'right_j1': -1.0258291091425074,
                            'right_j2': 0.0293680414401436,
                            'right_j3': 1.37518162913313,
                            'right_j4':  -0.06703022873354225,
                            'right_j5': 0.7968371433926965,
                            'right_j6': 1.7659649178699421}

fixed_orientation = Quaternion(
                         x=-0.00142460053167,
                         y=0.999994209902,
                         z=-0.00177030764765,
                         w=0.00253311793936)

from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_kinematics import KDLKinematics
import tf.transformations as tr
import PyKDL
from math import pi, cos, sin, radians
import tf_conversions.posemath as pm

from random import *
# to read ee position

from gym.utils import seeding
from gym import spaces
from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from intera_core_msgs.msg import JointCommand
from intera_core_msgs.msg import EndpointState
from intera_io import IODeviceInterface
import intera_interface
from intera_core_msgs.srv import (
    SolvePositionFK,
    SolvePositionFKRequest,
)   
from intera_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)

from gazebo_connection import GazeboConnection
from controllers_connection import ControllersConnection

overhead_orientation = Quaternion(
                            x=-0.00142460053167,
                            y=0.999994209902,
                            z=-0.00177030764765,
                            w=0.00253311793936)
# register(
#     id='Sawyer_DynaEnv-v0',
#     entry_point='sac.envs.robotics.reach:DyReachEnv',
#     max_episode_steps=200, #200
#     kwargs={'direction': (1, 0, 0), 'velocity': 0.011, 'distance_threshold':0.01, 'target_range': 0.10, 'test':False}
# )

ACTION_DIM = 8 # Cartesian
OBS_DIM = (100,100,3)      # POMDP
STATE_DIM = 24        # MDP
GRIPPER_UPPER = 0.041667
GRIPPER_LOWER = 0.0

TERM_THRES = 50
SUC_THRES = 50

CTRL_PERIOD = 70 # in Hz, hard sync 

class robotEnv(): 
    def __init__(self, max_steps=700, isdagger=False, isPOMDP=False, isGripper=False, isCartesian=True, train_indicator=1):
        """An implementation of OpenAI-Gym style robot reacher environment
        applicable to HER and HIRO.
        TODO: add method that receives target object's pose as state
        """
        rospy.init_node("robotEnv")
        # for compatiability
        self._limb = intera_interface.Limb("right")
        self.gripper =  intera_interface.Gripper("right") # should we instantiate gripper?
        if train_indicator:
            self._tip_name = 'right_gripper_tip'
        else:
            self._tip_name = 'right_hand'
        self.train_indicator = train_indicator # 1: Train 0:Test
        self.isdagger = isdagger
        self.isPOMDP = isPOMDP
        self.isReal = not train_indicator
        self.isGripper = isGripper
        self.isCartesian = isCartesian
        self.reached = False
        self.done = False
        print("Getting robot state... ")
        self._rs = intera_interface.RobotEnable(intera_interface.CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()
        self.bridge = CvBridge()
        self.joint_names = ['right_j0', 'right_j1', 'right_j2', 'right_j3', 'right_j4', 'right_j5', 'right_j6']
        self.joint_positions = list()
        self.joint_velocities = list()
        self.joint_efforts = list()
        self.gripper_joints = list()
        self.color_image = np.ones((400,400,3))
        self.endpt_pos = list()
        self.endpt_ori = list()
        self.joint_commands = list()
        self.max_steps = max_steps
        self.reward = 0
        self.reward_rescale = 1.0
        self.reward_type = 'dense'
        self.destPos = np.array([0.5, 0.0, 0.0])
        self.joint_cmd_msg = JointCommand()
        if self.isGripper:
            global STATE_DIM
            STATE_DIM +=2 # add dimenstion for gripper joints
            
        # variable for random block pose
        self.block_pose = Pose()
        # joint command publisher
        self.jointCmdPub = rospy.Publisher('/robot/limb/right/joint_command', JointCommand, tcp_nodelay=True, queue_size=1)
        self.demo_target_pub = rospy.Publisher('/demo/target/', Pose, queue_size=1)
        self.resize_factor = 100/400.0
        self.resize_factor_real = 100/400.0
        self.tf_listenser = tf.TransformListener()

        rospy.Subscriber('/robot/joint_states', JointState , self.jointStateCB)
        rospy.Subscriber('/robot/limb/right/endpoint_state', EndpointState , self.endpoint_positionCB)
        if self.train_indicator: # train
            rospy.Subscriber("/camera/color/image_raw", Image, self.rgb_ImgCB)
        else:
            rospy.Subscriber("/dynamic_objects/camera/raw_image", Image, self.rgb_ImgCB)
        self.distance_threshold = 0.10 # threshold for episode success
        self.tic = 0.0
        self.toc = 0.0
        self.elapsed = 0.0  
        # self.initial_state = self.get_joints_states().copy()
        self._action_scale = 1.0

        #open manipulator statets
        self.moving_state = ""
        self.actuator_state = ""
        if rospy.has_param('vel_calc'):
            rospy.delete_param('vel_calc')
        self.init_robot_pose()
        if train_indicator:
            self._load_gazebo_models()
        self.ee_trans = list()
        self.ee_rot = list()
        self.termination_count = 0
        self.success_count = 0
        if train_indicator:
            rospy.on_shutdown(self._delete_gazebo_models)

        # goal related
        self._hover_distance = 0.1
        # self.controller_connection = ControllersConnection(namespace='robot',
        #  controllers_list=['right_joint_position_controller', 'right_joint_velocity_controller', 'right_joint_effort_controller'])
      

    def move_to_start(self, start_angles=None):
        print("Moving the {0} arm to start pose...".format('right'))
        if not start_angles:
            start_angles = dict(zip(self._joint_names, [0]*7))
        self._guarded_move_to_joint_position(start_angles)
        self.gripper_open()


    def _guarded_move_to_joint_position(self, joint_angles, timeout=5.0):
        if rospy.is_shutdown():
            return
        if joint_angles:
            self._limb.move_to_joint_positions(joint_angles,timeout=timeout)
        else:
            rospy.logerr("No Joint Angles provided for move_to_joint_positions. Staying put.")


    def init_robot_pose(self):
        starting_joint_angles['right_j0'] = np.random.uniform(-0.05, 0.05)
        starting_joint_angles['right_j1'] = np.random.uniform(-0.95, -0.85)
        starting_joint_angles['right_j2'] = np.random.uniform(-0.1, 0.1)
        starting_joint_angles['right_j3'] = np.random.uniform(1.6, 1.7)

        start_pose = [starting_joint_angles['right_j0'], starting_joint_angles['right_j1'],
        starting_joint_angles['right_j2'], starting_joint_angles['right_j3'],
        starting_joint_angles['right_j4'], starting_joint_angles['right_j5'],
        starting_joint_angles['right_j6']]
        self.move_to_start(starting_joint_angles)


    def observation_space(self):
        """
        Observation space.

        :return: gym.spaces
                    observation space
        """
        return gym.spaces.Box(
            -np.inf,
            np.inf,
            shape=self.get_observation().shape,
            dtype=np.float32)


    def rgb_ImgCB(self, data):
        self.rcvd_color = data  ## ROS default image
        self.cimg_tstmp = rospy.get_time()
        self.color_image = self.bridge.imgmsg_to_cv2(self.rcvd_color, "bgr8") # 640 * 480
    

    def jointStateCB(self,msg): # callback function for joint state readings
        
        self.joint_positions = [self._limb.joint_angle('right_j0'),
        self._limb.joint_angle('right_j1'),
        self._limb.joint_angle('right_j2'),
        self._limb.joint_angle('right_j3'),
        self._limb.joint_angle('right_j4'),
        self._limb.joint_angle('right_j5'),
        self._limb.joint_angle('right_j6')]

        self.joint_velocities = [self._limb.joint_velocity('right_j0'),
        self._limb.joint_velocity('right_j1'),
        self._limb.joint_velocity('right_j2'),
        self._limb.joint_velocity('right_j3'),
        self._limb.joint_velocity('right_j4'),
        self._limb.joint_velocity('right_j5'),
        self._limb.joint_velocity('right_j6')]
        self.squared_sum_vel = np.linalg.norm(np.array(self.joint_velocities))

        self.joint_efforts = [self._limb.joint_effort('right_j0'),
        self._limb.joint_effort('right_j1'),
        self._limb.joint_effort('right_j2'),
        self._limb.joint_effort('right_j3'),
        self._limb.joint_effort('right_j4'),
        self._limb.joint_effort('right_j5'),
        self._limb.joint_effort('right_j6')]
        self.squared_sum_eff = np.linalg.norm(np.array(self.joint_efforts))


    def endpoint_positionCB(self, msg):
        """ Discrepancy btwn sim and real.
            sim : tip of the gripper
            real : base of the gripper

            gripper_pos = np.array(self._limb.endpoint_pose()['position'])
            gripper_ori = np.array(self._limb.endpoint_pose()['orientation'])
            gripper_lvel = np.array(self._limb.endpoint_velocity()['linear'])
            gripper_avel = np.array(self._limb.endpoint_velocity()['angular'])
            gripper_force = np.array(self._limb.endpoint_effort()['force'])
            gripper_torque = np.array(self._limb.endpoint_effort()['torque'])
        """
        if self.isReal:
            try:
                (self.endpt_pos, self.endpt_ori) = self.tf_listenser.lookupTransform('/base','/right_gripper_tip', rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                pass
        else:
            _endpt_pose = msg.pose
            self.endpt_ori = [_endpt_pose.orientation.x, _endpt_pose.orientation.y, _endpt_pose.orientation.z, _endpt_pose.orientation.w]
            self.endpt_pos = [_endpt_pose.position.x, _endpt_pose.position.y, _endpt_pose.position.z]


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def get_gripper_position(self):
        return self.gripper.get_position()


    def get_joints_states(self):
        return self.joint_positions, self.joint_velocities, self.joint_efforts


    def get_color_observation(self):
        if self.isReal:
            return cv2.resize(self.color_image, None, fx=self.resize_factor_real, fy=self.resize_factor_real, interpolation=cv2.INTER_CUBIC)
        else:
            return cv2.resize(self.color_image, None, fx=self.resize_factor, fy=self.resize_factor, interpolation=cv2.INTER_CUBIC)


    def get_end_effector_pose(self):
        return self.endpt_ori + self.endpt_pos


    def get_target_obj_obs(self):
        """Return target object pose. Experimentally supports only position info."""
        if not self.isReal:
            rospy.wait_for_service('/gazebo/get_model_state')
            try:
                object_state_srv = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
                object_state = object_state_srv("block", "base")
                self.destPos = np.array([object_state.pose.position.x, object_state.pose.position.y, object_state.pose.position.z-0.884])
            except rospy.ServiceException as e:
                rospy.logerr("Spawn URDF service call failed: {0}".format(e))
        return self.destPos


    def set_gripper_position(self, pos):
        self.gripper.set_position(pos)


    def set_action(self, joint_commands=list(), gripper_command=None):
        self.joint_commands = joint_commands
        self.joint_cmd_msg.mode = 2
        self.joint_cmd_msg.names = self.joint_names
        self.joint_cmd_msg.velocity = self.joint_commands # dtype should be list()
        self.jointCmdPub.publish(self.joint_cmd_msg)
        self._limb.set_command_timeout(1.0)
        if int(gripper_command)==1 and self._get_dist() <= self.distance_threshold:
            self.gripper_close()
        elif self._get_dist() <= self.distance_threshold:
            self.gripper_open()


    def test_servo_vel(self, time_step):
        dynamic_pose = self.dynamic_object(step=time_step)
        dynamic_pose.position.z += 0.1
        rospy.set_param('vel_calc','true')
        self.servo_vel(dynamic_pose, num_wp=2)
        if rospy.has_param('vel_calc'):
            rospy.logwarn('Initialized pose.')
            rospy.delete_param('vel_calc')

    @rate_limited(CTRL_PERIOD)
    def step(self, action=None, time_step=0):#overriden function
        """
        Function executed each time step.
        Here we get the action execute it in a time step and retrieve the
        observations generated by that action.self.reward
        :param action:
        :return: obs, reward, done
        """
        # self.test_servo_vel(time_step)
        # self.set_gripper_position(np.random.self.rewardniform(0.0,0.041))
        self.prev_tic = self.tic
        self.tic = time.time()
        self.elapsed =  time.time()-self.prev_tic
        self.done = False
        if time_step == self.max_steps:
            self.done = True
        #set action
        # act = action.flatten().tolist()
        act = action
        self.set_action(joint_commands=act[:-1],gripper_command=act[-1])
        if not self.isReal:
            curDist = self._get_dist()
            self.reward = self._compute_reward()
            if self.reward==1.0:
                self.success_count +=1
                if self.success_count == SUC_THRES:
                    print ('======================================================')
                    print ('Succeeds current Episode : SUCCEEDED')
                    print ('======================================================')
                    self.success_count = 0
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
            print("Actions: ", act)
        # achieved goal should be equivalent to the step observation dict.
        goal_obs = dict()
        obs = dict() # consists of 1. meas : [q, qdot, tau]  2. aux : [kine_pose, obj_pos] 3. observation : color_obs
        aux = [] # end_effector pose + target_object position
        obs['meas_state'] = [_joint_pos, _joint_vels, _joint_effos] # default observations
        goal_obs['full_state'] = [_joint_pos, _joint_vels, _joint_effos] # default goal_observations
        obs['auxiliary'] = [_targ_obj_obs]
    

        if self.isGripper:
            obs['meas_state'].append([_gripper_pos])
            goal_obs['full_state'].append([_gripper_pos])
        if self.isCartesian:
            obs['auxiliary'].append(_ee_pose)
            goal_obs['full_state'].append(_ee_pose)
        if self.isPOMDP:
            obs['color_obs'] = _color_obs
            goal_obs['color_obs'] = _color_obs
        return {'observation': obs,'achieved_goal':goal_obs}, self.reward_rescale*self.reward, self.done


    def get_desired_goals(self):
        """ how to acquire desired goal observation in real robot test environment?
        """
        _color_obs = self.get_color_observation()
        _joint_pos, _joint_vels, _joint_effos = self.get_joints_states()
        _gripper_pos = self.get_gripper_position()
        _targ_obj_obs = self.get_target_obj_obs()
        _ee_pose =  self.get_end_effector_pose()

        des_goal = dict()
        des_goal['full_state'] = [_joint_pos, _joint_vels, _joint_effos]
        if self.isGripper:
            des_goal['full_state'].append([_gripper_pos])
        if self.isCartesian:
            des_goal['full_state'].append(_ee_pose)
        if self.isPOMDP:
            des_goal['color_obs'] = _color_obs
        return des_goal


    def gen_block_pose(self):
        b_pose =Pose()
        b_pose.position.x = np.random.uniform(0.45, .75)
        b_pose.position.y = np.random.uniform(-0.2, 0.33)
        b_pose.position.z = 0.00    
        b_pose.orientation = overhead_orientation
        return b_pose 


    def retract(self):
        """retract after grasping&releasing target object
        """
        current_pose = self._limb.endpoint_pose()
        ik_pose = Pose()
        ik_pose.position.x = current_pose['position'].x
        ik_pose.position.y = current_pose['position'].y
        ik_pose.position.z = current_pose['position'].z + self._hover_distance
        ik_pose.orientation.x = current_pose['orientation'].x
        ik_pose.orientation.y = current_pose['orientation'].y
        ik_pose.orientation.z = current_pose['orientation'].z
        ik_pose.orientation.w = current_pose['orientation'].w
        self._servo_to_pose(ik_pose)
        print ('=============== retracting... ===============')
        rospy.sleep(1.0)


    def _reset_gazebo(self):
        """ Initialize the robot to its random pose.
            1. load target block with its random pose
            2. servo the robot to the position (TODO: orientation?)
            3. the target object should be grippeds when observing desired goals
        """
        block_pose = self.gen_block_pose()
        self._delete_target_block()
        self._delete_table()
        _des_goal = self._reset_desired_goal(goal_pose=block_pose, block_pose=block_pose)

        starting_joint_angles['right_j0'] = np.random.uniform(-0.05, 0.05)
        starting_joint_angles['right_j1'] = np.random.uniform(-0.95, -0.85)
        starting_joint_angles['right_j2'] = np.random.uniform(-0.1, 0.1)
        starting_joint_angles['right_j3'] = np.random.uniform(1.6, 1.7)

        _start_angles = [starting_joint_angles['right_j0'], starting_joint_angles['right_j1'],
        starting_joint_angles['right_j2'], starting_joint_angles['right_j3'],
        starting_joint_angles['right_j4'], starting_joint_angles['right_j5'],
        starting_joint_angles['right_j6']]
        self._limb.set_joint_position_speed(1.0)
        self.gripper_open()
        self.retract()
        _ = self.move_to_start(starting_joint_angles)
        # rospy.set_param('vel_calc', 'start')
        # while not rospy.is_shutdown() and rospy.has_param('vel_calc'):
        #     print ('Pending...')
        #     pass
        print("Moving the right arm to start pose...")
        return _des_goal


    def _reset_real(self):
        """ 
            Reset the real-world env.
            Initialize the robot to its random pose.
            1. load target block with its random pose
            2. servo the robot to the position (TODO: orientation?)
            3. the target object should be grippeds when observing desired goals
        """
        block_pose = self.gen_block_pose()
        print ('block_pose', block_pose)
        print ('PLEASE LOCATE THE BLOCK AT DESIRABLE LOCATION.')
        self._servo_to_pose(block_pose)
        self.gripper_close()
        _des_goal = self.get_desired_goals()
        starting_joint_angles['right_j0'] = np.random.uniform(-0.05, 0.05)
        starting_joint_angles['right_j1'] = np.random.uniform(-1.05, -0.85)
        starting_joint_angles['right_j2'] = np.random.uniform(-0.1, 0.1)
        starting_joint_angles['right_j3'] = np.random.uniform(1.6, 1.7)
        start_pose = [starting_joint_angles['right_j0'], starting_joint_angles['right_j1'],
        starting_joint_angles['right_j2'], starting_joint_angles['right_j3'],
        starting_joint_angles['right_j4'], starting_joint_angles['right_j5'],
        starting_joint_angles['right_j6']]
        # _ = self.move_to_start_vel_command(start_pose)
        _ = self.move_to_start(starting_joint_angles)
        print("Moving the right arm to start pose...")
        self.gripper_open()
        return _des_goal


    def _guarded_move_to_joint_position(self, joint_angles, timeout=5.0):
        if rospy.is_shutdown():
            return
        if joint_angles:
            self._limb.move_to_joint_positions(joint_angles,timeout=timeout)
        else:
            rospy.logerr("No Joint Angles provided for move_to_joint_positions. Staying put.")        


    def fk_service_client(self, joint_pose, limb = "right"):
        # returned value contains PoseStanped
        # std_msgs/Header header
        # geometry_msgs/Pose pose

          ns = "ExternalTools/" + limb + "/PositionKinematicsNode/FKService"
          fksvc = rospy.ServiceProxy(ns, SolvePositionFK)
          fkreq = SolvePositionFKRequest()
          joints = JointState()
          joints.name = ['right_j0', 'right_j1', 'right_j2', 'right_j3',
                         'right_j4', 'right_j5', 'right_j6']
          # joints.position = [0.763331, 0.415979, -1.728629, 1.482985,t
          #                    -1.135621, -1.674347, -0.496337]
          joints.position = joint_pose
          # Add desired pose for forward kinematics
          fkreq.configuration.append(joints)
          # Request forward kinematics from base to "right_hand" link
          # fkreq.tip_names.append('right_hand')
          fkreq.tip_names.append('right_gripper_tip')
          try:
              rospy.wait_for_service(ns, 5.0)
              resp = fksvc(fkreq)
          except (rospy.ServiceException, rospy.ROSException), e:
              rospy.logerr("Service call failed: %s" % (e,))
              return False

          # Check if result valid
          if (resp.isValid[0]):
              rospy.loginfo("SUCCESS - Valid Cartesian Solution Found")
              rospy.loginfo("\nFK Cartesian Solution:\n")
              rospy.loginfo("------------------")
              rospy.loginfo("Response Message:\n%s", resp)
              # print resp.pose_stamp[0].pose
              return resp.pose_stamp[0].pose
          else:
              rospy.logerr("INVALID JOINTS - No Cartesian Solution Found.")
              return False


    def gripper_open(self):
        self.gripper.open()


    def gripper_close(self):
        self.gripper.close()


    def _get_tf_matrix(self, pose):
        if isinstance(pose, dict):
            _quat = np.array([pose['orientation'].x, pose['orientation'].y, pose['orientation'].z, pose['orientation'].w])
            _trans = np.array([pose['position'].x, pose['position'].y, pose['position'].z])
            return tr.compose_matrix(angles=tr.euler_from_quaternion(_quat,'sxyz'), translate=_trans)
        else:
            _quat = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
            _trans = np.array([pose.position.x, pose.position.y, pose.position.z])
            return tr.compose_matrix(angles=tr.euler_from_quaternion(_quat,'sxyz'), translate=_trans)


    def _get_msg(self, tf_mat):
        _kdl_fr = pm.fromMatrix(tf_mat)
        _msg = pm.toMsg(_kdl_fr)

        return _msg


    def _get_pose(self, quat, trans):
        pose = Pose()
        pose.position.x = trans[0]
        pose.position.y = trans[1]
        pose.position.z = trans[2]
        pose.orientation.x = quat[0]
        pose.orientation.y = quat[1]
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]
        return pose
    

    def _check_traj(self, _pose, idx):
        if _pose: # if Forward Kinematics solution exists
            r = rospy.Rate(100) # command for 30Hz
            while not rospy.is_shutdown():
                _current_pose = self._limb.endpoint_pose()
                current_pose = np.array((_current_pose['position'].x, _current_pose['position'].y))
                target_pose = np.array(( _pose.position.x, _pose.position.y))
                _err = np.linalg.norm(current_pose-target_pose)

                _err_2 = (_pose.position.z - _current_pose['position'].z)

                if _err <= 0.10 and _err_2<=0.17: # if reached for target object in 5cm
                    rospy.loginfo("Reached waypoint {0}".format(idx))
                    break
                r.sleep()


    def servo_vel(self, pose, time=4.0, steps=400, num_wp=5):
        """Servo the robot to the goal
            TODO: merge check_path_stop method
        """
        current_pose = self._limb.endpoint_pose()
        X_current = self._get_tf_matrix(current_pose)
        X_goal = self._get_tf_matrix(pose)


        traj = mr.CartesianTrajectory(X_current, X_goal, Tf=5, N=num_wp, method=3)

        for idx, X in enumerate(traj):
            _pose = self._get_msg(X)
            self.ikVelPub.publish(_pose)
            self._check_traj(_pose, idx)


    def _reset_desired_goal(self, goal_pose=Pose(), block_pose=Pose()):
        """Manipulate the robot to the randomly generated target object.
            the orientation of the e,e, should be varied while satisfying the position.
            TODO: Use Sawyer velocity controller class 
        """
        # random position // quaterion from RPY
        # _ori_x = np.random.uniform(3.05,3.20)
        # _ori_y = np.random.uniform(-0.35,0.45)
        # _ori_z = np.random.uniform(-0.1,0.1)
        # if _ori_x >= pi:
        #     _ori_x -=2*pi
        # _rot = PyKDL.Rotation.RPY(_ori_x, _ori_y, _ori_z)
        # quat = _rot.GetQuaternion() # -> x, y, z, w
        _start_pose = goal_pose
        _start_pose.position.z += 0.00
        _start_pose.orientation = overhead_orientation
        self.gripper_open()
        # self._servo_to_pose(_start_pose) # des goal is not necessary
        self._load_table()
        rospy.logwarn('======================================================')
        print ('block pose', block_pose)
        rospy.logwarn('======================================================')
        self.demo_target_pub.publish(block_pose) # publish target pose for velocity controller
        self._load_target_block(block_pose=block_pose) 
        # rospy.sleep(1.0)
        # self.gripper_close()
        return self.get_desired_goals()


    def _delete_target_block(self):
        # This will be called on ROS Exit, deleting Gazebo models
        # Do not wait for the Gazebo Delete Model service, since
        # Gazebo should already be running. If the service is not
        # available since Gazebo has been killed, it is fine to error out
        try:
            delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
            resp_delete = delete_model("block")
        except rospy.ServiceException, e:
            rospy.loginfo("Delete Model service call failed: {0}".format(e))


    def _delete_table(self):
        # This will be called on ROS Exit, deleting Gazebo models
        # Do not wait for the Gazebo Delete Model service, since
        # Gazebo should already be running. If the service is not
        # available since Gazebo has been killed, it is fine to error out
        try:
            delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
            resp_delete = delete_model("cafe_table")
        except rospy.ServiceException as e:
            print("Delete Model service call failed: {0}".format(e))


    def _load_table(self):
        # This will be called on ROS Exit, deleting Gazebo models
        # Do not wait for the Gazebo Delete Model service, since
        # Gazebo should already be running. If the service is not
        # available since Gazebo has been killed, it is fine to error out
        # Get Models' Path
        table_pose=Pose(position=Point(x=0.75, y=0.0, z=0.0))
        table_reference_frame="world"
        model_path = rospkg.RosPack().get_path('sawyer_sim_examples')+"/models/"
        # Load Table SDF
        table_xml = ''
        with open (model_path + "cafe_table/model.sdf", "r") as table_file:
            table_xml=table_file.read().replace('\n', '')
        # Spawn Table SDF
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        try:
            spawn_sdf = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
            resp_sdf1 = spawn_sdf("cafe_table", table_xml, "/",
                                table_pose, table_reference_frame)
        except rospy.ServiceException, e:
            rospy.logerr("Spawn SDF service call failed: {0}".format(e)) 


    def _delete_gazebo_models(self):
        # This will be called on ROS Exit, deleting Gazebo models
        # Do not wait for the Gazebo Delete Model service, since
        # Gazebo should already be running. If the service is not
        # available since Gazebo has been killed, it is fine to error out
        try:
            delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
            resp_delete = delete_model("cafe_table")
            resp_delete = delete_model("block")
        except rospy.ServiceException as e:
            print("Delete Model service call failed: {0}".format(e))


    def reset(self):
        # Desired goal is given here.
        # Should wait until reset finishes.
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
        obs['meas_state'] = [_joint_pos, _joint_vels, _joint_effos] # default observations
        obs['auxiliary'] = [_targ_obj_obs] # default observations
        if self.isGripper:
            obs['meas_state'].append([_gripper_pos])
        if self.isCartesian:
            obs['auxiliary'].append(_ee_pose)
        if self.isPOMDP:
            obs['color_obs'] = _color_obs
        return {'observation': obs,'desired_goal':des_goal}


    def _load_target_block(self, block_pose=Pose(position=Point(x=0.6725, y=0.1265, z=0.7825)),
                        block_reference_frame="base"):
        # Get Models' Path
        model_path = rospkg.RosPack().get_path('sawyer_sim_examples')+"/models/"
        # Load Block URDF
        block_xml = ''
        num = randint(1,3)
        num = str(num)
        with open (model_path + "block/model_"+num+".urdf", "r") as block_file:
            block_xml=block_file.read().replace('\n', '')        
        # Spawn Block URDF
        rospy.wait_for_service('/gazebo/spawn_urdf_model')
        try:
            spawn_urdf = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
            resp_urdf = spawn_urdf("block", block_xml, "/",
                                block_pose, block_reference_frame)
        except rospy.ServiceException as e:
            rospy.logerr("Spawn URDF service call failed: {0}".format(e))


    def _load_gazebo_models(self, table_pose=Pose(position=Point(x=0.75, y=0.0, z=0.0)),
                        table_reference_frame="world",
                        block_pose=Pose(position=Point(x=0.4225, y=0.1265, z=1.1725)),
                        block_reference_frame="world",
                        kinect_pose=Pose(position=Point(x=1.50, y=0.0, z=1.50)),
                        kinect_reference_frame="world"
                        ):
        kinect_RPY = PyKDL.Rotation.RPY(0.0, 0.7854, 3.142)
        kinect_quat = kinect_RPY.GetQuaternion()
        kinect_pose.position.x = 1.50
        kinect_pose.position.y = 0.0
        kinect_pose.position.z = 1.50
        kinect_pose.orientation.x = kinect_quat[0]
        kinect_pose.orientation.y = kinect_quat[1]
        kinect_pose.orientation.z = kinect_quat[2]
        kinect_pose.orientation.w = kinect_quat[3]
        # Get Models' Path
        model_path = rospkg.RosPack().get_path('sawyer_sim_examples')+"/models/"
        # Load Table SDF
        table_xml = ''
        with open (model_path + "cafe_table/model.sdf", "r") as table_file:
            table_xml=table_file.read().replace('\n', '')
        #Load kinect SDF
        kinect_xml = ''
        with open (model_path + "kinect/model.sdf", "r") as kinect_file:
            kinect_xml=kinect_file.read().replace('\n', '')
        # Load Block URDF
        block_xml = ''
        with open (model_path + "block/model_1.urdf", "r") as block_file:
            block_xml=block_file.read().replace('\n', '')
        # Spawn Table SDF
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        try:
            spawn_sdf = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
            resp_sdf1 = spawn_sdf("cafe_table", table_xml, "/",
                                table_pose, table_reference_frame)
        except rospy.ServiceException, e:
            rospy.logerr("Spawn SDF service call failed: {0}".format(e))    


    def _check_for_termination(self):
        """
        Check if the agent has reached undesirable state. If so, terminate the episode early. 
        """
        if self.endpt_pos[0] < 0.34 or abs(self.endpt_pos[1])>0.33 or self.endpt_pos[2]<-0.1 or self.endpt_pos[2]>0.75 or self.endpt_pos[0] > 0.95:
            self.termination_count +=1
            if self.termination_count == TERM_THRES:
                self.termination_count =0
                return True
            else:
                return False    


    def _compute_reward(self):
        """Computes shaped/sparse reward for each episode.
        """
        cur_dist = self._get_dist()
        if self.reward_type == 'sparse':
            return (cur_dist <= self.distance_threshold).astype(np.float32) # 1 for success else 0
        else:
            return -cur_dist -self.squared_sum_vel # -L2 distance -l2_norm(joint_vels)


    def compute_reward_from_goal(self, achieved_goal, desired_goal):
        """Re-computed rewards for substituted goals. Only supports sparse reward setting.
        Computes batch array of rewards"""
        batch_dist = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return (batch_dist <= self.distance_threshold).astype(np.float32)


    def env_goal_transition(self, obs, next_obs, sub_goal):
        """ goal transition based on generic goal representation.
        h(s,g,s') = s + g - s' (all dimesion must match)
        """
        return obs + sub_goal - next_obs 


    def compute_intrinsic_reward(self, obs, next_obs, goal):
        """intrinsic reward for the low-level agent. based on the goal_transition
        """
        return -np.linalg.norm(self.env_goal_transition(obs, next_obs, goal))


    def _get_dist(self):
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            object_state_srv = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            # object_state = object_state_srv("block", "world")
            object_state = object_state_srv("block", "base")
            self._obj_pose = np.array([object_state.pose.position.x, object_state.pose.position.y, object_state.pose.position.z])
        except rospy.ServiceException as e:
            rospy.logerr("Spawn URDF service call failed: {0}".format(e))        
        _ee_pose = np.array(self.endpt_pos) # FK state of robot
        return np.linalg.norm(_ee_pose-self._obj_pose)


    def dynamic_object(self, pose = Pose(), step=0.0):
        """ dynamical movement of target object
        """
        step = 0
        delta = 2*pi*step/500.0
        center_x = 0.55
        radius = 0.15
        obj_pose = Pose()
        obj_pose.position.x = radius*cos(delta) + center_x       
        obj_pose.position.y = radius*sin(delta)
        obj_pose.position.z = 0.0
        obj_pose.orientation = overhead_orientation
        set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        state = ModelState()
        state.model_name ="block"
        state.pose = obj_pose
        state.reference_frame = "base"
        try:              
            resp = set_model_state(state)
            print (resp.status_message)                
        except Exception as e:            
            rospy.logerr('Error on calling service: %s',str(e))
        return obj_pose


    def _servo_to_pose(self, pose, time=4.0, steps=400.0):
        ''' An *incredibly simple* linearly-interpolated Cartesian move '''
        # r = rospy.Rate(1/(time/steps)) # Defaults to 100Hz command rate
        r = rospy.Rate(100.0) # Defaults to 100Hz command rate
        current_pose = self._limb.endpoint_pose()
        ik_delta = Pose()
        ik_delta.position.x = (current_pose['position'].x - pose.position.x) / steps
        ik_delta.position.y = (current_pose['position'].y - pose.position.y) / steps
        ik_delta.position.z = (current_pose['position'].z - pose.position.z) / steps
        ik_delta.orientation.x = (current_pose['orientation'].x - pose.orientation.x) / steps
        ik_delta.orientation.y = (current_pose['orientation'].y - pose.orientation.y) / steps
        ik_delta.orientation.z = (current_pose['orientation'].z - pose.orientation.z) / steps
        ik_delta.orientation.w = (current_pose['orientation'].w - pose.orientation.w) / steps
        for d in range(int(steps), -1, -1):
            if rospy.is_shutdown():
                return
            ik_step = Pose()
            ik_step.position.x = d*ik_delta.position.x + pose.position.x
            ik_step.position.y = d*ik_delta.position.y + pose.position.y
            ik_step.position.z = d*ik_delta.position.z + pose.position.z
            ik_step.orientation.x = d*ik_delta.orientation.x + pose.orientation.x
            ik_step.orientation.y = d*ik_delta.orientation.y + pose.orientation.y
            ik_step.orientation.z = d*ik_delta.orientation.z + pose.orientation.z
            ik_step.orientation.w = d*ik_delta.orientation.w + pose.orientation.w
            joint_angles = self._limb.ik_request(ik_step, self._tip_name)
            if joint_angles:
                self._limb.set_joint_positions(joint_angles)
            else:
                rospy.logerr("No Joint Angles provided for move_to_joint_positions. Staying put.")
            r.sleep()


    def close(self):        
        rospy.signal_shutdown("done")


    def fk_service_client(self, joint_pose, limb = "right"):
        # returned value contains PoseStanped
        # std_msgs/Header header
        # geometry_msgs/Pose pose
        ns = "ExternalTools/" + limb + "/PositionKinematicsNode/FKService"
        fksvc = rospy.ServiceProxy(ns, SolvePositionFK)
        fkreq = SolvePositionFKRequest()
        joints = JointState()
        joints.name = ['right_j0', 'right_j1', 'right_j2', 'right_j3',
                         'right_j4', 'right_j5', 'right_j6']

        joints.position = joint_pose
          # Add desired pose for forward kinematics
        fkreq.configuration.append(joints)
          # Request forward kinematics from base to "right_hand" link
        # fkreq.tip_names.append('right_hand')
        fkreq.tip_names.append('right_gripper_tip')

        try:
            rospy.wait_for_service(ns, 5.0)
            resp = fksvc(fkreq)
        except (rospy.ServiceException, rospy.ROSException), e:
            rospy.logerr("Service call failed: %s" % (e,))
            return False

          # Check if result valid
        if (resp.isValid[0]):
            # rospy.loginfo("SUCCESS - Valid Cartesian Solution Found")
            # rospy.loginfo("\nFK Cartesian Solution:\n")
            # rospy.loginfo("------------------")
            # rospy.loginfo("Response Message:\n%s", resp)
              # print resp.pose_stamp[0].pose
            return resp.pose_stamp[0].pose
        else:
            # rospy.logerr("INVALID JOINTS - No Cartesian Solution Found.")
            return False