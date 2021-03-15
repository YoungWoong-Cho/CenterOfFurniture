# Copyright 2021 by YoungWoon Cho
# The Cooper Union for the Advancement of Science and Art
# ECE471 Machine Learning Architecture
# Project 2: Discovery

import argparse
from torch.utils.data import DataLoader
from Dataset import FurnitureDataset
from models import DeepAE
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

def main():
	# Define arguments
	parser = argparse.ArgumentParser(description="This is a pytorch implementation of VAE. Please refer to the following arguments.")
	parser.add_argument('--batch_size', default=1, type=int, help='Size of a mini-batch. Default: 1')
	parser.add_argument('--cuda', action="store_true", help="Turn on the cuda option.")
	parser.add_argument('--dataset', nargs='+', type=str, default=['tables', 'chairs'], help='Datasets to train. Possible choices: beds, chairs, dressers, lamps, sofas, tables. Default: tables chairs')
	parser.add_argument('--epochs', default=10, type=int, help='Epochs to run. Default: 10')
	parser.add_argument('--encoding_dim', default=32, type=int, help='Encoding dimension. Default: 32')
	parser.add_argument('--img_size', default=128, type=int, help='Image size in pixels. Default: 128')
	args = parser.parse_args()

	# Device configuration
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Create dataloader objects
	img_size = args.img_size
	furniture_dataloader = []
	for furniture_type in args.dataset:
		dataset = FurnitureDataset(furniture_type, img_size)
		dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle=True)
		furniture_dataloader.append(dataloader)

	# Prepare model
	model = DeepAE(encoding_dim=args.encoding_dim, img_size=img_size)#.to(device)

	# Optimizer
	optimizer = torch.optim.Adam(model.parameters())

	# Train
	for epoch in range(0, args.epochs):
		for dataloader in furniture_dataloader:
			#print(type(dataloader))
			progress = tqdm(enumerate(dataloader), total=len(dataloader))
			for i, data in progress:
				# Forward pass
				data = data['item']#.to(device)
				data = data.view(data.shape[0], -1)
				data_reconst = model(data)
				loss = F.binary_cross_entropy(data_reconst, data)

				# Backward and optimize
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				# Verbose
				progress.set_description(
                f'[{epoch}/{args.epochs - 1}][{i}/{len(dataloader) - 1}]'
                f'Loss: {loss:.4f} ')


if __name__ == "__main__":
    main()