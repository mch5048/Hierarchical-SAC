#!/usr/bin/env python
#-----------------------------------------------------------------------
# Stripped down version of velocity_controller.py
# Runs at 100Hz
#
# Tasks:
# 1. Finds Jb and Vb
# 2. Uses naive least-squares damping to find qdot
#   (Addresses inf vel at singularities)
# 3. Publishes qdot to Sawyer using the limb interface
#
# Written By Stephanie L. Chang
# Last Updated: 4/13/17
#-----------------------------------------------------------------------
# Python Imports
import numpy as np
from math import pow, pi, sqrt
import tf.transformations as tr
import threading
import tf

# ROS Imports
import rospy
from std_msgs.msg import Bool, Int32, Float64
from geometry_msgs.msg import Pose, Point, Quaternion
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_kinematics import KDLKinematics
import intera_interface
from intera_core_msgs.msg import EndpointState

# Local Imports
import sawyer_MR_description as s
import modern_robotics as r
import custom_logging as cl

####################
# GLOBAL VARIABLES #
####################
TIME_LIMIT = 7    #15s
DAMPING = 0.06
JOINT_VEL_LIMIT = 2    #2rad/s

class VelocityControl(object):
    def __init__(self):
        rospy.loginfo("Creating VelocityController class")

        # Create KDL model
        with cl.suppress_stdout_stderr():    # Eliminates urdf tag warnings
            self.robot = URDF.from_parameter_server()

        # self.kin = KDLKinematics(self.robot, "base", "right_gripper")
        self.kin= KDLKinematics(self.robot, "base", "right_gripper_tip")

        self.names = self.kin.get_joint_names()

        self.limb = intera_interface.Limb("right")
        self.gripper = intera_interface.gripper.Gripper('right')
        # Grab M0 and Blist from saywer_MR_description.py
        self.M0 = s.M #Zero config of right_hand
        self.Blist = s.Blist #6x7 screw axes mx of right arm
        self.Kp = 50*np.eye(6)
        self.Ki = 0.0*np.eye(6)
        self.it_count = 0
        self.int_err = 0
        # Shared variables
        self.mutex = threading.Lock()
        self.time_limit = rospy.get_param("~time_limit", TIME_LIMIT)
        self.damping = rospy.get_param("~damping", DAMPING)
        self.joint_vel_limit = rospy.get_param("~joint_vel_limit", JOINT_VEL_LIMIT)
        self.q = np.zeros(7)        # Joint angles
        self.qdot = np.zeros(7)     # Joint velocities
        self.T_goal = np.array(self.kin.forward(self.q))    # Ref se3
        self.original_goal = self.T_goal.copy()    # Ref se3
        # print self.T_goal
        self.isCalcOK = False
        self.isPathPlanned = False
        self.traj_err_bound = float(1e-2) # in meter
        # Subscriber
        self.ee_position = [0.0, 0.0, 0.0]
        self.ee_orientation = [0.0, 0.0, 0.0, 0.0]

        rospy.Subscriber('/demo/target/', Pose, self.ref_poseCB)
        rospy.Subscriber('/robot/limb/right/endpoint_state', EndpointState , self.endpoint_poseCB)

        self.gripper.calibrate()
        # path planning
        self.num_wp = int(5)
        self.cur_wp_idx = int(0) # [0:num_wp - 1]
        self.traj_list = [None for _ in range(self.num_wp)]

        self.r = rospy.Rate(100)
        # control loop
        while not rospy.is_shutdown():
            if rospy.has_param('vel_calc'):
                if not self.isPathPlanned: # if path is not planned
                    self.path_planning() # get list of planned waypoints
                self.calc_joint_vel_2()
            self.r.sleep()


    def ref_poseCB(self, goal_pose): # Takes target pose, returns ref se3
        rospy.logdebug("ref_pose_cb called in velocity_control.py")
        # p = np.array([some_pose.position.x, some_pose.position.y, some_pose.position.z])
        p = np.array([goal_pose.position.x, goal_pose.position.y, goal_pose.position.z])
        quat = np.array([goal_pose.orientation.x, goal_pose.orientation.y, goal_pose.orientation.z, goal_pose.orientation.w])
        goal_tmp = tr.compose_matrix(angles=tr.euler_from_quaternion(quat, 'sxyz'), translate=p) # frame is spatial 'sxyz', return Euler angle from quaternion for specified axis sequence.
        with self.mutex:
            self.original_goal = goal_tmp


    def endpoint_poseCB(self, ee_pose):
        self.ee_position = [ee_pose.position.x, ee_pose.position.y, ee_pose.position.z]
        self.ee_orientation = [ee_pose.orientation.x, ee_pose.orientation.y, ee_pose.orientation.z, ee_pose.orientation.w]


    def _get_ee_position(self):
        return self.ee_position


    def _get_goal_matrix(self):
        return self.traj_list[self.cur_wp_idx]


    def path_planning(self, num_wp=self.num_wp):
        """ Generate desriable waypoints for achieving target pose.
        """
        current_pose = self._limb.endpoint_pose()
        X_current = self._get_tf_matrix(current_pose)
        X_goal = self.original_goal
        self.traj_list = mr.CartesianTrajectory(X_current, X_goal, Tf=5, N=num_wp, method=3)
        # self.T_goal = self.traj_list[self.cur_wp_idx] # set T_goal as the starting waypoint (robot's current pose)
        self.isPathPlanned = True


    def _check_traj(self):
        """Check if the end-effector has reached the desired position of target waypoint.
        """
        _ee_position = self._get_ee_position()
        _targ_wp_position = tf.translation_from_matrix(self.traj_list[self.cur_wp_idx])
        if np.linal.norm( np.array(_ee_position) -_targ_wp_position) <= self.traj_err_bound:
            rospy.loginfo('reached waypoint %d', self.cur_wp_idx)
            if self.cur_wp_idx < self.num_wp - 1: # indicate next waypoint
                self.cur_wp_idx += 1
            elif self.cur_wp_idx == self.num_wp - 1: # robot has reached the last waypoint
                self.cur_wp_idx = 0
                rospy.delete_param('vel_ctrl')
                self.isPathPlanned = False


    def _get_tf_matrix(self, pose):
        """Return the homogeneous matrix of given Pose
        """
        if isinstance(pose, dict):
            _quat = np.array([pose['orientation'].x, pose['orientation'].y, pose['orientation'].z, pose['orientation'].w])
            _trans = np.array([pose['position'].x, pose['position'].y, pose['position'].z])
            return tr.compose_matrix(angles=tr.euler_from_quaternion(_quat,'sxyz'), translate=_trans)
        else:
            _quat = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
            _trans = np.array([pose.position.x, pose.position.y, pose.position.z])
            return tr.compose_matrix(angles=tr.euler_from_quaternion(_quat,'sxyz'), translate=_trans)


    def get_q_now(self):         # Finds current joint angles
        qtemp = np.zeros(7)
        i = 0
        while i<7:
            qtemp[i] = self.limb.joint_angle(self.names[i])
            i += 1
        with self.mutex:
            self.q = qtemp              # Angles in radians

    def stop_oscillating(self):
        i = 0
        v_norm = 0
        qvel = self.qdot

        while i<7:
            v_norm += pow(qvel[i],2)
            i += 1
        v_norm = sqrt(v_norm)

        if v_norm < 0.1:
            self.qdot = np.zeros(7)
        return

    def calc_joint_vel(self):
        rospy.logdebug("Calculating joint velocities...")

        # Body stuff
        Tbs = self.M0 # 
        Blist = self.Blist # 

        # Current joint angles
        self.get_q_now()
        with self.mutex:
            q_now = self.q #1x7 mx

        # Desired config: base to desired - Tbd
        with self.mutex:
            T_sd = self.T_goal # 

        # Find transform from current pose to desired pose, error
        # refer to CH04 >> the product of exponential formula for open-chain manipulator
        # sequence:
        #   1. FKinBody (M, Blist, thetalist)
        #       M: the home config. of the e.e.
        #       Blist: the joint screw axes in the ee frame, when the manipulator is @ the home pose
        #       thetalist : a list of current joints list
        #       We get the new transformation matrix T for newly given joint angles
        #   
        #           1). np.array(Blist)[:,i] >> retrieve one axis' joint screw 
        #           2). ex)  [s10, -c10, 0., -1.0155*c10, -1.0155*s10, -0.1603] -> S(theta)
        #           3). _out = VecTose3(_in)
        #                 # Takes a 6-vector (representing a spatial velocity).
        #                 # Returns the corresponding 4x4 se(3) matrix.
        #           4). _out = MatrixExp6(_in)
        #                    # Takes a se(3) representation of exponential coordinate 
        #                    # Returns a T matrix SE(3) that is achieved by traveling along/about the
        #                    # screw axis S for a distance theta from an initial configuration T = I(dentitiy)
        #           5). np.dot (M (from base to robot's e.e. @home pose , and multiplying exp coord(6), we get new FK pose
        #
        #   2. TransInv >> we get the inverse of homogen TF matrix
        #   3. error = np.dot (T_new, T_sd) >> TF matrix from cur pose to desired pose
        #   4. Vb >>compute the desired body twist vector from the TF matrix 
        #   5. JacobianBody:
        #           # In: Blist, and q_now >> Out : T IN SE(3) representing the end-effector frame when the joints are
        #                 at the specified coordinates


        e = np.dot(r.TransInv(r.FKinBody(Tbs, Blist, q_now)), T_sd) # Modern robotics pp 230 22Nff

        # Desired TWIST: MatrixLog6 SE(3) -> se(3) exp coord
        Vb = r.se3ToVec(r.MatrixLog6(e))  # shape : (6,)
        # Construct BODY JACOBIAN for current config
        Jb = r.JacobianBody(Blist, q_now) #6x7 mx # shape (6,7)
        # WE NEED POSITION FEEDBACK CONTROLLER

        # Desired ang vel - Eq 5 from Chiaverini & Siciliano, 1994
        # Managing singularities: naive least-squares damping
        n = Jb.shape[-1] #Size of last row, n = 7 joints
        # OR WE CAN USE NUMPY' PINV METHOD 

        invterm = np.linalg.inv(np.dot(Jb.T, Jb) + pow(self.damping, 2)*np.eye(n)) # 
        
        
        
        qdot_new = np.dot(np.dot(invterm,Jb.T),Vb) # It seems little bit akward...? >>Eq 6.7 on pp 233 of MR book

        self.qdot = qdot_new #1x7

        # Constructing dictionary
        qdot_output = dict(zip(self.names, self.qdot))

        # Setting Sawyer right arm joint velocities
        self.limb.set_joint_velocities(qdot_output)
        # print qdot_output
        return

    def scale_joint_vel(self, q_dot):
        minus_v = abs(np.amin(q_dot))
        plus_v = abs(np.amax(q_dot))
        if minus_v > plus_v:
            scale = minus_v
        else:
            scale = plus_v
        if scale > self.joint_vel_limit:
            return 1.0*(q_dot/scale)*self.joint_vel_limit


    def calc_joint_vel_2(self):
        # self.it_count += 1
        # print self.it_count
        # self.cur_theta_list = np.delete(self.main_js.position, 1)
        Tbs = self.M0 # 
        Blist = self.Blist # 
        self.get_q_now()
        with self.mutex:
            q_now = self.q #1x7 mx
        with self.mutex:
            # T_sd = self.T_goal # 
            T_sd = self._get_goal_matrix() # 
            # print T_sd

        e = np.dot(r.TransInv(r.FKinBody(Tbs, Blist, q_now)), T_sd) # Modern robotics pp 230 22Nff
        Xe = r.se3ToVec(r.MatrixLog6(e))
        # self.X_d = self.tf_lis.fromTranslationRotation(self.p_d, self.Q_d)
        # self.X = mr.FKinBody(self.M, self.B_list, self.cur_theta_list)
        # self.X_e = mr.se3ToVec(mr.MatrixLog6(np.dot(mr.TransInv(self.X), self.X_d)))
        if np.linalg.norm(self.int_err) < 10 and np.linalg.norm(self.int_err) > -10:
            # self.int_err = (self.int_err + self.X_e)
            self.int_err = (self.int_err + Xe)

        Vb = np.dot(self.Kp, Xe) + np.dot(self.Ki, self.int_err)

        # self.V_b = np.dot(self.Kp, self.X_e) + np.dot(self.Ki, self.int_err)
        self.J_b = r.JacobianBody(Blist, self.q)
        n = self.J_b.shape[-1] #Size of last row, n = 7 joints

        invterm = np.linalg.inv(np.dot(self.J_b.T, self.J_b) + pow(self.damping, 2)*np.eye(n)) # 

        self.q_dot = np.dot(np.dot(invterm, self.J_b.T),Vb) # It seems little bit akward...? >>Eq 6.7 on pp 233 of MR book
        self.q_dot = self.scale_joint_vel(self.q_dot)
        qdot_output = dict(zip(self.names, self.q_dot))

        self.arm.set_joint_velocities(qdot_output)
        # self.stop_oscillating()
        self._check_traj()


def main():
    rospy.init_node('velocity_control')

    try:
        vc = VelocityControl()
    except rospy.ROSInterruptException:
        pass

    rospy.spin()

if __name__ == '__main__':
    main()
