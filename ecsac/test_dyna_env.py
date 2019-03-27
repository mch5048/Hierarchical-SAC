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
from SetupSummary import SummaryManager_SAC as SummaryManager
import pickle
import os
import import_intera
# apply hindsight experience replay.


if __name__ == '__main__':
    env = robotEnv()
    _ = env.reset()
    for i in range(2000):
        _, _, _ = env.step(time_step=i)
        