#!/usr/bin/env python
import sys
import os
import rospy
import numpy as np
# reads open manipulator's state
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import *
from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from gazebo_msgs.msg import ContactsState
import math
import cv2
from cv_bridge import CvBridge, CvBridgeError
from string import Template
import time
from ddpg.msg import GoalObs
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from gazebo_msgs.srv import (
    GetModelState
)
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

from math import pow, pi, sqrt
import tf.transformations as tr
import tf

import rospy
from std_msgs.msg import Bool, Int32, Float64
from geometry_msgs.msg import Pose, Point, Quaternion
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_kinematics import KDLKinematics
import PyKDL
import os


os.chdir("/home/irobot/catkin_ws/src/ddpg/scripts")
rospy.init_node('pose_acquisition')

limb = intera_interface.Limb("right")
head_display = intera_interface.HeadDisplay()
head_display.display_image('/home/irobot/Downloads' + "/Cute2.png")
cuff = intera_interface.Cuff()
tf_listenser = tf.TransformListener()

endpt_ori = [0.0, 0.0, 0.0, 0.0]
endpt_pos = [0.0, 0.0, 0.0]
position_x = list()
position_y = list()
position_z = list()
orient_x = list()
orient_y = list()
orient_z = list()
orient_w = list()
roll = list()
pitch = list()
yaw = list()

joint_p = [list() for p in range(7)]
joint_v = [list() for v in range(7)]
joint_e = [list() for e in range(7)]

joint_names = ['right_j0', 'right_j1', 'right_j2', 'right_j3', 'right_j4', 'right_j5', 'right_j6']

def append_position_to_list():
    global cuff
    global endpt_ori
    global endpt_pos
    if cuff.lower_button():
        try:
            (endpt_pos, endpt_ori) = tf_listenser.lookupTransform('/base','/right_gripper_tip', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            pass

        global position_x
        global position_y
        global position_z
        global orient_x
        global orient_y
        global orient_z
        global orient_w
        global limb
        global roll
        global pitch
        global yaw
        global joint_p
        global joint_v
        global joint_e


        # dict of joint angles
        joint_angles = limb.joint_angles()
        joint_velocities = limb.joint_velocities()
        joint_efforts = limb.joint_efforts()

        for idx in range(7):
            joint_p[idx].append(joint_angles[joint_names[idx]])

        for idx in range(7):
            joint_v[idx].append(joint_velocities[joint_names[idx]])

        for idx in range(7):
            joint_e[idx].append(joint_efforts[joint_names[idx]])

        current_pose = limb.endpoint_pose()
        # position_x.append(current_pose['position'].x)
        # position_y.append(current_pose['position'].y)
        # position_z.append(current_pose['position'].z)
        position_x.append(endpt_pos[0])
        position_y.append(endpt_pos[1])
        position_z.append(endpt_pos[2])
        # qx = current_pose['orientation'].x
        # qy = current_pose['orientation'].y
        # qz = current_pose['orientation'].z
        # qw = current_pose['orientation'].w
        orient_x.append(endpt_ori[0])
        orient_y.append(endpt_ori[1])
        orient_z.append(endpt_ori[2])
        orient_w.append(endpt_ori[3])
        _rot = PyKDL.Rotation.Quaternion(*endpt_ori)
        RPY = _rot.GetRPY()
        roll.append(RPY[0])
        pitch.append(RPY[1])
        yaw.append(RPY[2])

        rospy.loginfo("logging...")

def save_pose_to_csv():
    global position_x
    global position_y
    global position_z
    global orient_x
    global orient_y
    global orient_z
    global orient_w
    global roll
    global pitch
    global yaw
    global joint_p
    global joint_v
    global joint_e

    position_npar_x = np.array(position_x)
    position_npar_y = np.array(position_y)
    position_npar_z = np.array(position_z)
    orient_npar_x = np.array(orient_x)
    orient_npar_y = np.array(orient_y)
    orient_npar_z = np.array(orient_z)
    orient_npar_w = np.array(orient_w)
    roll_npar = np.array(roll)
    pitch_npar = np.array(pitch)
    yaw_npar = np.array(yaw)

    joint_p_npar = np.array(joint_p)
    joint_v_npar = np.array(joint_v)
    joint_e_npar = np.array(joint_e)


    data_save_arr = np.column_stack((position_npar_x,
                                    position_npar_y,
                                    position_npar_z,
                                    orient_npar_x,
                                    orient_npar_y,
                                    orient_npar_z,
                                    orient_npar_w,
                                    roll_npar,
                                    pitch_npar,
                                    yaw_npar,
                                    np.array(joint_p[0]),
                                    np.array(joint_p[1]),
                                    np.array(joint_p[2]),
                                    np.array(joint_p[3]),
                                    np.array(joint_p[4]),
                                    np.array(joint_p[5]),
                                    np.array(joint_p[6]),
                                    np.array(joint_v[0]),
                                    np.array(joint_v[1]),
                                    np.array(joint_v[2]),
                                    np.array(joint_v[3]),
                                    np.array(joint_v[4]),
                                    np.array(joint_v[5]),
                                    np.array(joint_v[6]),
                                    np.array(joint_e[0]),
                                    np.array(joint_e[1]),
                                    np.array(joint_e[2]),
                                    np.array(joint_e[3]),
                                    np.array(joint_e[4]),
                                    np.array(joint_e[5]),
                                    np.array(joint_e[6])
                                                                       
                                    ))


    # if os.path.isfile("sawyer_ee_pose.csv"):
    #     os.remove("sawyer_ee_pose.csv")

    np.savetxt("sawyer_ee_pose_{0}.csv".format(os.getpid()), data_save_arr, delimiter=',', fmt='%.3e')
    position_x = list()
    position_y = list()
    position_z = list()
    orient_x = list()
    orient_y = list()
    orient_z = list()
    orient_w = list()
    roll = list()
    pitch = list()
    yaw = list()    
    joint_p = [list() for p in range(7)]
    joint_v = [list() for v in range(7)]
    joint_e = [list() for e in range(7)]

    rospy.sleep(1.0)
    rospy.logwarn("saved joint pose")


def main():
    r = rospy.Rate(100)
    while not rospy.is_shutdown():
        append_position_to_list()
        if cuff.upper_button():
            save_pose_to_csv()


if __name__ == '__main__':
    main()