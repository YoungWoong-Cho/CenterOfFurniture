# Copyright 2021 by YoungWoong Cho
# The Cooper Union for the Advancement of Science and Art
# ECE471 Machine Learning Architecture
# Project 2: Discovery
"""
This is a class where the deep AE architecture is realized.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Fully connected neural network with one hidden layer
class Encoder(nn.Module):
    def __init__(self, encoding_dim, img_size):
        super().__init__()
        self.input_shape = [3, img_size, img_size]
        self.input_size = np.prod(self.input_shape)

        self.model = nn.Sequential(
            nn.Linear(self.input_size, encoding_dim*4),
            nn.ReLU(),
            nn.Linear(encoding_dim*4, encoding_dim*2),
            nn.ReLU(),
            nn.Linear(encoding_dim*2, encoding_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self, encoding_dim, img_size):
        super().__init__()
        self.input_shape = [3, img_size, img_size]
        self.input_size = np.prod(self.input_shape)

        self.model = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim*2),
            nn.ReLU(),
            nn.Linear(encoding_dim*2, encoding_dim*4),
            nn.ReLU(),
            nn.Linear(encoding_dim*4, self.input_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

class VAE_encoder(nn.Module):
	# |-------- ENCODER --------||REPARAMETRIZE||------------ DECODER ------------|
	#					  -- fc2 ----- mu ----
	#  x -- fc1 -- relu -|			          |- z -- fc4 -- relu -- fc5 -- sigmoid
	#					  -- fc3 ----- std ---
	# |-------- ENCODER --------||REPARAMETRIZE||------------ DECODER ------------|
	# x: input
	# fc: fully connected layer
	# relu: activation function
	# mu: mean
	# std: standard deviation
	# eps: randomly generated from gaussian distribution
	# z = mu + std*eps
	def __init__(self, encoding_dim, img_size):
		super().__init__()
		self.input_shape = [3, img_size, img_size]
		self.input_size = np.prod(self.input_shape)

		self.fc1 = nn.Linear(self.input_size, encoding_dim*2)
		self.fc2 = nn.Linear(encoding_dim*2, encoding_dim)
		self.fc3 = nn.Linear(encoding_dim*2, encoding_dim)

		self.N = torch.distributions.Normal(0, 1)
		self.N.loc = self.N.loc.cuda()
		self.N.scale = self.N.scale.cuda()
		self.kl_div = 0

	def forward(self, x):
		x = torch.flatten(x, start_dim=1)
		x = F.relu(self.fc1(x))
		mu = self.fc2(x)
		sigma = torch.exp(self.fc3(x))
		z = mu + sigma*self.N.sample(mu.shape)
		self.kl_div = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
		return z

class VAE_decoder(nn.Module):
	# Takes in z and returns reconstructed_x
	def __init__(self, encoding_dim, img_size):
		super().__init__()
		self.input_shape = [3, img_size, img_size]
		self.input_size = np.prod(self.input_shape)

		self.model = nn.Sequential(
			nn.Linear(encoding_dim, encoding_dim*2),
			nn.ReLU(),
			nn.Linear(encoding_dim*2, self.input_size),
			nn.Sigmoid()
		)

	def forward(self, z):
		return self.model(z)

class ConvAE_encoder(nn.Module):
	def __init__(self, encoding_dim):
		super().__init__()
		self.model = nn.Sequential(
			# [3, 64, 64]
			nn.Conv2d(3, 16, 3, stride= 1, padding= 1),
			# [16, 64, 64]
			nn.ReLU(),
			nn.MaxPool2d(2),
			# [16, 32, 32]
			nn.Conv2d(16, 16, 3, stride= 1, padding= 1),
			# [16, 32, 32]
			nn.ReLU(),
			nn.Flatten(),
			# [1, 16*32*32]
			nn.ReLU(),
			nn.Linear(16*32*32, encoding_dim),
			# [1, encoding_dim]
			nn.ReLU()
		)

	def forward(self, x):
		return self.model(x)

class ConvAE_decoder(nn.Module):
	def __init__(self, encoding_dim):
		super().__init__()
		self.model = nn.Sequential(
			nn.Linear(encoding_dim, 16*32*32),
			nn.ReLU(),
			nn.Unflatten(1, (16, 32, 32)),
			# [16, 32, 32]
			nn.ReLU(),
			nn.Conv2d(16, 16, 3,stride= 1, padding= 1),
			# [16, 32, 32]
			nn.Upsample(scale_factor=2),
			# [16, 64, 64]
			nn.Conv2d(16, 3, 3,stride= 1, padding= 1),
			# [3, 64, 64]
			nn.ReLU(True),
			nn.Sigmoid()
		)

	def forward(self, x):
		return self.model(x)

class ConvAE(nn.Module):
	def __init__(self, encoding_dim):
		super().__init__()
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.encoder = ConvAE_encoder(encoding_dim).to(device)
		self.decoder = ConvAE_decoder(encoding_dim).to(device)
	def forward(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded