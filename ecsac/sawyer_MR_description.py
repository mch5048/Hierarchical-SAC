#-----------------------------------------------------------------------
# Tasks:
# 1. Returns M0 matrix and body twists for Sawyer
#
# Obtained from Chainatee Tanakulrungson
#-----------------------------------------------------------------------

import numpy as np
from math import cos, sin, radians

s10 = sin(radians(10))
c10 = cos(radians(10))

# DH matrix
# Blist: The joint screw axes in the end-effector frame when the 
# manipulator is at the home position,


# Blist = np.array([[s10, -c10, 0., -1.0155*c10, -1.0155*s10, -0.1603],
#                   [-c10, -s10, 0., -0.9345*s10, 0.9345*c10, 0.],
#                   [0. , 0., 1., -0.0322*s10, 0.0322*c10, 0.],
#                   [-c10, -s10, 0., -0.5345*s10, 0.5345*c10, 0.],
#                   [0., 0., 1., 0.1363*s10, -0.1363*c10, 0.],
#                   [-c10, -s10, 0., -0.1345*s10, 0.1345*c10, 0.],
#                   [0., 0., 1., 0., 0., 0.]])


Blist = np.array([[s10, -c10, 0., -1.15112*c10, -1.15112*s10, -0.1603],
                  [-c10, -s10, 0., -0.9345*s10, 0.9345*c10, 0.],
                  [0. , 0., 1., -0.0322*s10, 0.0322*c10, 0.],
                  [-c10, -s10, 0., -0.5345*s10, 0.5345*c10, 0.],
                  [0., 0., 1., 0.1363*s10, -0.1363*c10, 0.],
                  [-c10, -s10, 0., -0.1345*s10, 0.1345*c10, 0.],
                  [0., 0., 1., 0., 0., 0.]])




Blist = Blist.T


# Homogen TF matrix from base to robot's end-effector

# M: The home configuration (position and orientation) of the 
#         end-effector
# Should make it automated
# use tf.transformation to change from ee_pose + quaternion to homogen tf 


M = np.array([[0., 0., 1., 1.15112],
              [-c10, -s10, 0., 0.1603],
              [s10, -c10, 0., 0.317],
              [0., 0., 0., 1.]])


# M = np.array([[0., 0., 1., 1.0155],
#               [-c10, -s10, 0., 0.1603],
#               [s10, -c10, 0., 0.317],
#               [0., 0., 0., 1.]])
