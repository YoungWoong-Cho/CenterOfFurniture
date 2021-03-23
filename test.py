from dataset import FurnitureDataset
from models import Encoder, Decoder
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import argparse
import os
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd

def main():
	# Define arguments
	parser = argparse.ArgumentParser(description="This is a pytorch implementation of VAE. Please refer to the following arguments.")
	parser.add_argument('--encoding_dim', default=32, type=int, help='Encoding dimension. Default: 32')
	parser.add_argument('--img_size', default=128, type=int, help='Image size in pixels. Default: 128')
	parser.add_argument('--mode', default='encode', type=str, help='Encode or Decode. Default: encode')
	parser.add_argument('--model', default='tables_chairs', type=str, help='Model to use. Models can be found under \'./weights/\' directory. Default: beds_chairs')
	args = parser.parse_args()

	# Device configuration
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Prepare image
	furnitures = args.model.split('_')
	furniture_dataloader = []
	for furniture_type in furnitures:
		dataset = FurnitureDataset(furniture_type, args.img_size)
		dataloader = DataLoader(dataset)
		furniture_dataloader.append(dataloader)

	# Load model
	if args.mode=='encode':
		model = Encoder(encoding_dim=args.encoding_dim, img_size=args.img_size).to(device)
	elif args.mode=='decode':
		model = Decoder(encoding_dim=args.encoding_dim, img_size=args.mg_size).to(device)
	model_path = os.path.join('./weights', f'{args.model}_{args.mode}r.pth')
	model.load_state_dict(torch.load(model_path))

	# Create latent space
	# Create dictionary for each furniture type so that the latent vector distribution can be analyzed separately
	latent_vector = [{'vector': [], 'label': []} for _ in range(len(furnitures))]
	
	for furniture_idx, dataloader in enumerate(furniture_dataloader):
		progress = tqdm(enumerate(dataloader), total=len(dataloader))
		for i, data in progress:
			# Encode
			furniture_item = data['item'].to(device)
			furniture_item = furniture_item.view(furniture_item[0], -1)
			furniture_item_encoded = model(furniture_item)

			# Populate latent space
			furniture_type = data['type'][0]
			latent_vector[furniture_idx]['vector'].append(furniture_item_encoded.cpu().detach().numpy())
			latent_vector[furniture_idx]['label'].append(furniture_type)

			# Verbose
			progress.set_description(
			f'[{i}/{len(dataloader) - 1}]')

	# Visualize latent space using t-SNE
	# vstack to convert from dictionary to feature matrix
	# latent_vector[0] is a list of latent vectors of tables, etc..
	latent_vector = [np.vstack(latent_vector[i]['vector']) for i in range(len(furnitures))]
	latent_vector = [pd.DataFrame(data=latent_vector[i]) for i in range(len(furnitures))]

	fig,ax = render_mpl_table
	latent_vector[0].describe(include='all')
	latent_vector[1].describe(include='all')

	# tsne_model = TSNE(learning_rate=100)
	# tsne_transformed = tsne_model.fit_transform(latent_space_vector)

	# # Plot
	# colors = [f'C{n}' for n in range(len(furnitures))]
	# colors_dict = {furnitures[i]:colors[i] for i in range(len(furnitures))}
	# c=[]
	# for label in latent_vector['label']: c.append(colors_dict[label])
	# plt.figure(figsize=(6, 6))
	# plt.scatter(tsne_transformed[:, 0], tsne_transformed[:, 1], s=2, c=c)
	# plt.show()

if __name__=='__main__':
	main()