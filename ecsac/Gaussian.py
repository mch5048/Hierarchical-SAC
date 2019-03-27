#!/usr/bin/env python

import random
import numpy as np 

class Gaussian(object):
	'''
		an implementation of naive Gaussian noise for action DDPG
	'''
	def __init__(self, is_uniform=False, mu=0.0, sigma=1.0):
		self.mu = mu
		self.sigma = sigma


	def __call__(self):
		return np.random.normal(self.mu, self.sigma)


	# def reset(self):
		# self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

# class NormalActionNoise(ActionNoise):
#     def __init__(self, mu, sigma):
#         self.mu = mu
#         self.sigma = sigma

#     def __call__(self):
#         return np.random.normal(self.mu, self.sigma)

#     def __repr__(self):
#         return 'NormalActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)