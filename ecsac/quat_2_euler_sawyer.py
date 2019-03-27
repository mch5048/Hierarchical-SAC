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

os.chdir("/home/irobot/catkin_ws/src/ddpg/scripts")

rospy.init_node('pose_acquisition')

limb = intera_interface.Limb("right")
head_display = intera_interface.HeadDisplay()
head_display.display_image('/home/irobot/Downloads' + "/Cute.png")
cuff = intera_interface.Cuff()

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

def append_position_to_list():
    global cuff
    if cuff.lower_button():
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
        current_pose = limb.endpoint_pose()
        position_x.append(current_pose['position'].x)
        position_y.append(current_pose['position'].y)
        position_z.append(current_pose['position'].z)
        qx = current_pose['orientation'].x
        qy = current_pose['orientation'].y
        qz = current_pose['orientation'].z
        qw = current_pose['orientation'].w
        orient_x.append(qx)
        orient_y.append(qy)
        orient_z.append(qz)
        orient_w.append(qw)
        _rot = PyKDL.Rotation.Quaternion(qx,qy,qz,qw)

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

    data_save_arr = np.column_stack((position_npar_x,
                                    position_npar_y,
                                    position_npar_z,
                                    orient_npar_x,
                                    orient_npar_y,
                                    orient_npar_z,
                                    orient_npar_w,
                                    roll_npar,
                                    pitch_npar,
                                    yaw_npar))

    if os.path.isfile("sawyer_ee_pose.csv"):
        os.remove("sawyer_ee_pose.csv")
    np.savetxt("sawyer_ee_pose.csv", data_save_arr, delimiter=',', fmt='%.3e')

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
    rospy.sleep(1.0)


def main():
    r = rospy.Rate(100)
    while not rospy.is_shutdown():
        append_position_to_list()
        if cuff.upper_button():
            save_pose_to_csv()


if __name__ == '__main__':
    main()