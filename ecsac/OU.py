#!/usr/bin/env python

import random
import numpy as np 

class OU(object):
	def __init__(self, mu, sigma, theta=.015, dt=1e-2, x0=None):
		self.mu = mu
		self.sigma = sigma
		self.dt = dt
		self.x0 = x0
		self.theta = theta
		self.reset()

	def __call__(self):
		x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
		self.x_prev = x
		return x

	def reset(self):
		self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

