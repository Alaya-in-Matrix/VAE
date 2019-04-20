from VAE import VAE,Encoder,Decoder
import torch
import torch.nn as nn
import pyro
import pyro.optim
import torchvision as vision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np

# Trained with 4000 hentai images from https://github.com/alexkimxyz/nsfw_data_scraper

img_size = 128
use_cuda = True
transf   = transforms.Compose([
    transforms.Resize((img_size,img_size)), 
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
images     = datasets.ImageFolder('./data/Hentai/', transform=transf)
loader     = torch.utils.data.DataLoader(dataset = images, pin_memory=use_cuda, shuffle=True, batch_size=1)
conf       = dict()
conf['lr'] = 1e-3
vae        = VAE(n_channel=3,img_size=img_size,z_dim = 1024, use_cuda = use_cuda, conf = conf)
num_epochs = 30
for epoch in range(num_epochs):
    loss = vae.one_epoch(loader)
    print('Epoch %3d, loss = %g' % (epoch, loss), flush = True)

torch.save(vae, 'saved_vae')
