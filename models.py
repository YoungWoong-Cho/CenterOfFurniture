import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Fully connected neural network with one hidden layer
class DeepAE(nn.Module):
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
            nn.ReLU(),

            nn.Linear(encoding_dim, encoding_dim*2),
            nn.ReLU(),
            nn.Linear(encoding_dim*2, encoding_dim*4),
            nn.ReLU(),
            nn.Linear(encoding_dim*4, self.input_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)