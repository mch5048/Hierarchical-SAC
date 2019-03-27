# -*- coding: utf-8 -*-
"""DDPG agent for episodic tasks in OpenAI Gym.

- Author: Curt Park
- Contact: curt.park@medipixel.io
- Paper: https://arxiv.org/pdf/1509.02971.pdf
"""

import argparse
import os
from typing import Tuple

import gym
import numpy as np
import torch
import torch.nn.functional as F
import wandb

import algorithms.common.helper_functions as common_utils
from algorithms.common.abstract.agent import AbstractAgent
from algorithms.common.buffer.replay_buffer import HER_ReplayBuffer
from algorithms.common.noise import OUNoise
from her import Her_sampler
from normalizer import RunningMeanStd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent(AbstractAgent):
    """ActorCritic interacting with environment.

    Attributes:
        memory (ReplayBuffer): replay memory
        noise (OUNoise): random noise for exploration
        hyper_params (dict): hyper-parameters
        actor (nn.Module): actor model to select actions
        actor_target (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        critic_target (nn.Module): target critic model to predict state values
        actor_optimizer (Optimizer): optimizer for training actor
        critic_optimizer (Optimizer): optimizer for training critic
        curr_state (np.ndarray): temporary storage of the current state

    """

    def __init__(
        self,
        env: gym.Env,
        args: argparse.Namespace,
        hyper_params: dict,
        models: tuple,
        optims: tuple,
        noise: OUNoise,
    ):
        """Initialization.

        Args:
            env (gym.Env): openAI Gym environment with discrete action space
            args (argparse.Namespace): arguments including hyperparameters and training settings
            hyper_params (dict): hyper-parameters
            models (tuple): models including actor and critic
            optims (tuple): optimizers for actor and critic
            noise (OUNoise): random noise for exploration
        
        Crucial attributes:
            reward_ftn (method):
                Example :
                def goal_distance(goal_a, goal_b):
                    assert goal_a.shape == goal_b.shape
                    return np.linalg.norm(goal_a - goal_b, axis=-1)
                    
                def compute_reward(self, achieved_goal, goal, info):
                    # Compute distance between goal and the achieved goal.
                    d = goal_distance(achieved_goal, goal)
                    if self.reward_type == 'sparse':
                        return -(d > self.distance_threshold).astype(np.float32)
                    else:
                        return -d



        """
        AbstractAgent.__init__(self, env, args)

        self.actor, self.actor_target, self.critic, self.critic_target = models
        self.actor_optimizer, self.critic_optimizer = optims
        self.hyper_params = hyper_params
        self.curr_state = np.zeros((1,))
        self.noise = noise
        # get an environment's reward function : sparse / dense 
        self.reward_ftn = env.reward_ftn(reward_type='sparse')

        # load the optimizer and model parameters
        if args.load_from is not None and os.path.exists(args.load_from):
            self.load_params(args.load_from)

        #obs_normalizer
        self.obs_norm = RunningMeanStd(shape=(1,) + env.obs_shape) #
        self.goal_norm = RunningMeanStd(shape=(1,) + env.goal_shape) # 

        #HER
        self.her_sampler = Her_sampler(reward_func=self.reward_ftn)
        
        # replay memory
        self.memory = HER_ReplayBuffer(
            hyper_params["BUFFER_SIZE"], hyper_params["BATCH_SIZE"], self.args.seed, normalizer=[self.obs_norm, self.goal_norm],
            her_sampler=self.her_sampler,
        reward_ftn = self.reward_ftn)
        self.ep_obs, self.ep_obs_1, self.ep_ag, self.ep_ag_1, self.ep_g, self.ep_act, self.ep_rew, self.ep_dn = [], [], [], [], [], [], [] , [] 

    def _concat_obs(self, obs, g):
        n_obs = self.obs_norm.normalize(obs)
        n_g = self.goal_norm.normalize(g)
        # concatenate the stuffs
        inputs = np.concatenate([n_obs, n_g])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
        return inputs

    def select_action(self, obs: np.ndarray, goal: np.ndarray) -> torch.Tensor:
        """Select an action from the input space."""
        self.curr_obs = obs
        self.des_goal = goal

        obs = torch.FloatTensor(obs).to(device)
        goal = torch.FloatTensor(goal).to(device)
        _act_input = _concat_obs(obs, goal)
        selected_action = self.actor()
        selected_action += torch.FloatTensor(self.noise.sample()).to(device)

        # TODO: Define action limit here
        selected_action = torch.clamp(selected_action, -1.0, 1.0)

        return selected_action

    def step(self, action: torch.Tensor) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env.
           Since we will apply HER, reward/done at this time is not important.
           Achieved goal for next state will be appended at the end of each epsode.
        """
        action = action.detach().cpu().numpy()
        next_observation, reward, done, _ = self.env.step(action) # next obs is tuple

        next_obs = next_observation['observation']
        achvd_goal = next_observation['achieved_goal']

        self.ep_obs.append(self.curr_state.copy())
        self.ep_obs_1.append(next_obs)
        self.ep_g.append(self.des_goal.copy())
        self.ep_actions.append(action) 
        self.ep_rews.append(reward)
        self.ep_dns.append(done)

        return next_observation, reward, done

    def append_ag_terminal(self, ag_terminal):
        """Append achieved goal @ terminal state."""
        self.ep_ag_1 = self.ep_ag[1:]
        self.ep_ag_1.append(ag_terminal)

    def update_model(
        self,
        experiences: Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        , torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Train the model after each episode."""
        obs, acts, rews, obs_nxt, goals, dones = experiences

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        masks = 1 - dones
        # next_actions = self.actor_target(next_states)
        # next_values = self.critic_target(torch.cat((next_states, next_actions), dim=-1))
        # curr_returns = rewards + self.hyper_params["GAMMA"] * next_values * masks
        # curr_returns = curr_returns.to(device)

        next_actions = self.actor_target(obs_nxt) # a'
        next_values = self.critic_target(torch.cat((obs_nxt, next_actions), dim=-1)) # target Q
        curr_returns = rewards + self.hyper_params["GAMMA"] * next_values * masks
        curr_returns = curr_returns.to(device)

        # train critic
        values = self.critic(torch.cat((obs, acts), dim=-1))
        critic_loss = F.mse_loss(values, curr_returns)
        self.critic_optimizer.zero_grad() # all grads of all params to zero
        critic_loss.backward() # compute gradients
        self.critic_optimizer.step() # apply gradients

        # train actor
        actions = self.actor(obs)
        actor_loss = -self.critic(torch.cat((obs, acts), dim=-1)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update target networks
        tau = self.hyper_params["TAU"]
        common_utils.soft_update(self.actor, self.actor_target, tau)
        common_utils.soft_update(self.critic, self.critic_target, tau)

        return actor_loss.data, critic_loss.data

    def load_params(self, path: str):
        """Load model and optimizer parameters."""
        if not os.path.exists(path):
            print("[ERROR] the input path does not exist. ->", path)
            return

        params = torch.load(path)
        self.actor.load_state_dict(params["actor_state_dict"])
        self.actor_target.load_state_dict(params["actor_target_state_dict"])
        self.critic.load_state_dict(params["critic_state_dict"])
        self.critic_target.load_state_dict(params["critic_target_state_dict"])
        self.actor_optimizer.load_state_dict(params["actor_optim_state_dict"])
        self.critic_optimizer.load_state_dict(params["critic_optim_state_dict"])
        print("[INFO] loaded the model and optimizer from", path)

    def save_params(self, n_episode: int):
        """Save model and optimizer parameters."""
        params = {
            "actor_state_dict": self.actor.state_dict(),
            "actor_target_state_dict": self.actor_target.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "actor_optim_state_dict": self.actor_optimizer.state_dict(),
            "critic_optim_state_dict": self.critic_optimizer.state_dict(),
        }

        AbstractAgent.save_params(self, self.args.algo, params, n_episode)

    def write_log(self, i: int, loss: np.ndarray, score: int):
        """Write log about loss and score"""
        total_loss = loss.sum()

        print(
            "[INFO] episode %d total score: %d, total loss: %f\n"
            "actor_loss: %.3f critic_loss: %.3f\n"
            % (i, score, total_loss, loss[0], loss[1])  # actor loss  # critic loss
        )

        if self.args.log:
            wandb.log(
                {
                    "score": score,
                    "total loss": total_loss,
                    "actor loss": loss[0],
                    "critic loss": loss[1],
                }
            )
    def _normalize_obs(self, obs, g_obs):
        obs_norm = self.obs_norm.normalize(obs)
        g_norm = self.goal_norm.normalize(g)
        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
        return inputs

    def update_normalizers(self, batch_obs):
        """Update mean / stddev of normaizers from batch observations."""
        ep_obs, ep_goal = batch_obs
        nd_obs = np.array(ep_obs)
        nd_goal = np.array(ep_goal)
        self.obs_norm.update(nd_obs)
        self.goal_norm.update(nd_goal)
        
    def train(self):
        """Train the agent."""
        # logger
        if self.args.log:
            wandb.init()
            wandb.config.update(self.hyper_params)
            wandb.watch([self.actor, self.critic], log="parameters")

        for i_episode in range(1, self.args.episode_num + 1):
            observation = self.env.reset() # now state is tuple
            obs = observation['observation']
            g = observation['desired_goal']
            done = False
            score = 0
            loss_episode = list()
            ep_steps = 0
            # first loop of HER
            while not done:
                ep_steps += 1
                # if self.args.render and i_episode >= self.args.render_after:
                #     self.env.render()

                action = self.select_action(observation) # requires inspection
                next_observation, reward, done = self.step(action)
                observation = next_observation
                score += reward

            # second loop of HER
            self.append_ag_terminal(ag_terminal=observation['achieved_goal'])
            self.memory.add_episode_to_buffer([self.ep_obs, self.ep_obs_1, self.ep_ag, self.ep_ag_1, self.ep_g, self.ep_act, self.ep_rew, self.ep_dn])
            self.memory.execute_normalised_goal_strategy([self.ep_obs, self.ep_obs_1, self.ep_ag, self.ep_ag_1, self.ep_g, self.ep_act, self.ep_rew, self.ep_dn])
            self.update_normalizers([self.ep_obs, self.ep_g])

            # third loop of HER : train with normalized & substituted obs&goals
            if len(self.memory) >= self.hyper_params["BATCH_SIZE"]: # ensure sufficient traning data
                for _ in range(ep_steps): # train as much as rollout steps
                    experiences = self.memory.sample()
                    loss = self.update_model(experiences)
                    loss_episode.append(loss)  # for logging
                    
            # logging
            if loss_episode:
                avg_loss = np.vstack(loss_episode).mean(axis=0)
                self.write_log(i_episode, avg_loss, score)

            if i_episode % self.args.save_period == 0:
                self.save_params(i_episode)

        # termination
        self.env.close()
