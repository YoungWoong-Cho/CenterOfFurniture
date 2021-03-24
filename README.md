# Discovery: Center of Furniture

### TL;JUST RUN THE CODE
<ul>
<li>Pillow==8.1.2
<li>tensorflow==2.4.1
<li>torch==1.8.0+cu111
<li>torchvision==0.9.0+cu111
</ul>
Type in the following command:

    python train.py

If you want to train the model with your own model,

    python train.py --dataset furniture1 furniture2 --epochs 100 --img_size 128

Possible types of furnitures can be found by typing

    python train.py --help

After training, you can run the latent space intrapolation and latent space visualization by running latent_space_visualization.ipynb.

Run the code block under <strong>Create a latent space intrapolation gif</strong> to generate a latent space intrapolation gif animation.

Run the code block under <strong>Create a latent space visualization with t-SNE</strong> to generate a 2D latent space visualization.

For help, type

    python train.py --help
