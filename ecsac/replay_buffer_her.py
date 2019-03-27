# -*- coding: utf-8 -*-
"""Replay buffer for baselines."""

import random
from collections import deque

import numpy as np
import torch
import threading

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class HER_ReplayBuffer:
    """Fixed-size buffer to store experience tuples.

    Taken from Udacity deep-reinforcement-learning github repository:
    https://github.com/udacity/deep-reinforcement-learning/blob/master/
    ddpg-pendulum/ddpg_agent.py

    Extended version to be used with HER implementation

    Attributes:
        buffer (deque): deque of replay buffer
        batch_size (int): size of a batched sampled from replay buffer for training

    """

    def __init__(self, buffer_size, batch_size, seed, her_sampler=None, demo=None):
        """Initialize a ReplayBuffer object.

        Args:
            buffer_size (int): size of replay buffer for experience
            batch_size (int): size of a batched sampled from replay buffer for training
            seed (int): random seed
            her_sampler (object):
            demo (deque) : demonstration deque

        """
        self.buffer = deque(maxlen=buffer_size) if not demo else demo
        self.lock = threading.Lock()
        self.batch_size = batch_size
        self.her_sampler = her_sampler
        random.seed(seed)

    def add(self, state, action, reward, next_state, desired_goal, achvd_goal, done):
        """Add a new experience to memory."""
        self.buffer.append((state, action, reward, next_state, desired_goal, achvd_goal, done))

    def add_terminal_ag(self, terminal_ag):
        terminal_ag = terminal_ag

    def add_episode_to_buffer(self, episode_batch):
        """Add whole episodic transitions to batch."""
        ep_obs, ep_obs_1, ep_ag, ep_ag_1, ep_g, ep_act, ep_rew, ep_dn = episode_batch
        
        ep_obs = np.array(ep_obs)
        ep_obs_1 = np.array(ep_obs_1)
        ep_ag = np.array(ep_ag)
        ep_ag_1 = np.array(ep_ag_1)
        ep_g = np.array(ep_g)
        ep_act= np.array(ep_act)
        ep_rew = np.array(ep_rew)
        ep_dn = np.array(ep_dn)

        batch_size = ep_obs.shape[0]
        with self.lock:
            # store the informations
            for idx, data in enumerate(zip(ep_obs, ep_obs_1, ep_ag, ep_ag_1, ep_g, ep_act, ep_rew, ep_dn)):
                self.buffer.append(data)

    def execute_normalised_goal_strategy(self, episode_batch):
        """Normalise the observations batch by batch."""
        ep_obs, ep_obs_1, ep_ag, ep_ag_1, ep_g, ep_act, ep_rew, ep_dn = episode_batch
        
        # get the number of normalization transitions
        num_transitions = ep_obs.shape[0] # for all transitions 
        # HER substitution for all transitions
        transitions = self.her_sampler.sample_her_transitions(episode_batch, num_transitions)


    def extend(self, transitions):
        """Add experiences to memory."""
        self.buffer.extend(transitions)

    def sample(self):
        """Randomly sample a batch of experiences from memory.
           Transitions for HER are sampled together, 
           but not concatenated as described in the original paper.
        """
        experiences = random.sample(self.buffer, k=self.batch_size)

        # 0,   5,    6,     1,      4,     7    
        obs, acts, rews, obs_nxt, goals, dones = [], [], [], [], [], []

        # return necessary samples for traning
        for e in experiences:
            obs.append(np.expand_dims(e[0], axis=0))
            acts.append(e[5])
            rews.append(e[6])
            obs_nxt.append(np.expand_dims(e[1], axis=0))
            goals.append(np.expand_dims(e[4], axis=0))
            dones.append(e[7])

        # why vstack here?
        obs = torch.from_numpy(np.vstack(obs)).float().to(device)
        acts = torch.from_numpy(np.vstack(acts)).float().to(device)
        rews = torch.from_numpy(np.vstack(rews)).float().to(device)
        obs_nxt = torch.from_numpy(np.vstack(obs_nxt)).float().to(device)
        goals = torch.from_numpy(np.vstack(goals)).float().to(device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(device)

        return (obs, acts, rews, obs_nxt, goals, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.buffer)
