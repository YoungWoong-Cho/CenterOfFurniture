# Copyright 2021 by YoungWoong Cho
# The Cooper Union for the Advancement of Science and Art
# ECE471 Machine Learning Architecture
# Project 2: Discovery
"""
This is a helper class that handles the dataset.
"""

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

class FurnitureDataset(Dataset):
	"""Furniture Dataset"""
	def __init__(self, furniture_type, img_size):
		self.furniture_type = furniture_type
		self.img_size = img_size
		self.root_dir = './dataset/Bonn_Furniture_Styles_Dataset'
		if os.path.exists(os.path.join(self.root_dir, 'metadata', f'{furniture_type}.txt')):
			print(f'Loading {furniture_type} dataset...')
			self.dataset = pd.read_csv(os.path.join(self.root_dir, 'metadata', f'{furniture_type}.txt'), delimiter='\t', header=None)
			self.dataset = self.dataset.dropna(axis=1, how='all')
			self.dataset.columns = ['style', 'filename', 'metadata']

			print(f'{furniture_type} dataset loaded successfully.')
		else: print(f'ERROR: {furniture_type} dataset not found. Please check if the name is correct.')

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		image_path = os.path.join(self.root_dir, list(self.dataset['filename'])[idx])
		image = Image.open(image_path).resize((self.img_size, self.img_size))
		image = transforms.ToTensor()(image)
		sample = {'item': image, 'type': self.furniture_type}
		return sample