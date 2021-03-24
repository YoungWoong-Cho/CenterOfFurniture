# Copyright 2021 by YoungWoong Cho
# The Cooper Union for the Advancement of Science and Art
# ECE471 Machine Learning Architecture
# Project 2: Center of Furniture

from dataset import FurnitureDataset
from models import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

def main():
	print(torch.__version__)

	# Define arguments
	parser = argparse.ArgumentParser(description="This is a pytorch implementation of VAE. Please refer to the following arguments.")
	parser.add_argument('--batch_size', default=1, type=int, help='Size of a mini-batch. Default: 1')
	parser.add_argument('--dataset', nargs='+', type=str, default=['tables', 'chairs'], help='Datasets to train. Possible choices: beds, chairs, dressers, lamps, sofas, tables. Default: tables chairs')
	parser.add_argument('--encoding_dim', default=32, type=int, help='Encoding dimension. Default: 32')
	parser.add_argument('--epochs', default=10, type=int, help='Epochs to run. Default: 10')
	parser.add_argument('--img_size', default=256, type=int, help='Image size in pixels. Default: 256')
	args = parser.parse_args()

	# Device configuration
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Create dataloader objects
	img_size = args.img_size
	furniture_dataloader = []
	filename = ''
	for furniture_type in args.dataset:
		dataset = FurnitureDataset(furniture_type, img_size)
		dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle=True)
		furniture_dataloader.append(dataloader)
		filename = furniture_type if len(filename) ==0 else filename + '_' + furniture_type

	# Prepare model
	encoder = Encoder(encoding_dim=args.encoding_dim, img_size=img_size).to(device)
	decoder = Decoder(encoding_dim=args.encoding_dim, img_size=img_size).to(device)

	# Optimizer
	optimizer_encoder = torch.optim.Adam(encoder.parameters())
	optimizer_decoder = torch.optim.Adam(decoder.parameters())

	# Train
	for epoch in range(0, args.epochs):
		for dataloader in furniture_dataloader:
			progress = tqdm(enumerate(dataloader), total=len(dataloader))
			for i, data in progress:
				data = data['item'].to(device)
				data = data.view(data.shape[0], -1)

				# Forward pass
				data_encoded = encoder(data)
				data_decoded = decoder(data_encoded)

				# Find loss
				loss = F.mse_loss(data, data_decoded)

				# Back propagation and optimization
				optimizer_encoder.zero_grad()
				optimizer_decoder.zero_grad()
				loss.backward()
				optimizer_encoder.step()
				optimizer_decoder.step()

				# Verbose
				progress.set_description(
				f'[{epoch}/{args.epochs - 1}][{i}/{len(dataloader) - 1}]'
				f'Loss: {loss:.4f} ')

	torch.save(encoder.state_dict(), f'./weights/{filename}_encoder.pth')
	torch.save(decoder.state_dict(), f'./weights/{filename}_decoder.pth')

if __name__ == "__main__":
    main()