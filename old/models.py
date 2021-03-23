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
from torch.autograd import Variable

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
	def __init__(self, img_size, encoding_dim):
		super().__init__()
		self.input_shape = [3, img_size, img_size]
		self.input_size = np.prod(self.input_shape)

		self.fc1 = nn.Linear(self.input_size, encoding_dim*2)
		self.fc_mu = nn.Linear(encoding_dim*2, encoding_dim)
		self.fc_var = nn.Linear(encoding_dim*2, encoding_dim)

	def forward(self, x):
		h1 = self.fc1(x)
		return self.fc_mu(h1), self.fc_var(h1)

class VAE_decoder(nn.Module):
	# Takes in z and returns reconstructed_x
	def __init__(self, img_size, encoding_dim):
		super().__init__()
		self.input_shape = [3, img_size, img_size]
		self.input_size = np.prod(self.input_shape)

		self.fc1 = nn.Linear(encoding_dim, encoding_dim*2)
		self.fc2 = nn.Linear(encoding_dim*2, self.input_size)

		self.ReLU = nn.ReLU()  # reLU non-linear unit for the hidden output
		self.sigmoid = nn.Sigmoid()  # sigmoid non-linear unit for the output

	def forward(self, embedded):
		h1 = self.ReLU(self.fc1(embedded))
		return self.sigmoid(self.fc2(h1))

class VariationalAutoEncoder(nn.Module):
	def __init__(self, img_size, encoding_dim):
		super().__init__()
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.encoder = VAE_encoder(img_size, encoding_dim).to(self.device)
		self.decoder = VAE_decoder(img_size, encoding_dim).to(self.device)

	def forward(self, x):
		mu, log_var = self.encoder(x.view(x.shape[0], -1))
		z = self.reparameterize(mu, log_var)
		return self.decoder(z), mu, log_var

	def reparameterize(self, mu, log_var):
		vector_size = log_var.size()
		eps = Variable(torch.FloatTensor(vector_size).normal_()).to(self.device)
		std = log_var.mul(0.5).exp_().to(self.device)
		return eps.mul(std).add_(mu)

class VAELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCELoss()
        self.bce_loss.size_average = False

    def forward(self, x, mu, log_var, recon_x):
        batch_size = x.size()[0]
        BCE = self.bce_loss(recon_x, x)

        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_element = mu.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
        KLD = torch.sum(KLD_element).mul_(-0.5)

        return (BCE + KLD) / batch_size