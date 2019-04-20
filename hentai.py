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

img_size   = 128
batch_size = 32
use_cuda   = True
num_epochs = 100
lr         = 5e-4
z_dim      = 256
transf     = transforms.Compose([
    transforms.Resize((img_size,img_size)), 
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
images                    = datasets.ImageFolder('./data/Hentai/', transform=transf)
len_train                 = int(0.95 * len(images))
len_validate              = len(images) - len_train
train_imgs, validate_imgs = torch.utils.data.random_split(images, [len_train, len_validate])
train_loader              = torch.utils.data.DataLoader(dataset = train_imgs,    pin_memory=use_cuda, shuffle=True, batch_size=batch_size)
validate_loader           = torch.utils.data.DataLoader(dataset = validate_imgs, pin_memory=use_cuda, shuffle=True, batch_size=batch_size)
conf                      = dict()
conf['lr']                = lr
vae                       = VAE(n_channel=3,img_size=img_size,z_dim = z_dim, use_cuda = use_cuda, conf = conf)
for epoch in range(num_epochs):
    train_loss    = vae.one_epoch(train_loader)
    validate_loss = vae.evaluate(validate_loader)
    print('Epoch %3d, train_loss = %11.2f valid_loss = %11.2f' % (epoch, train_loss, validate_loss), flush = True)
vae.eval()
torch.save(vae, 'saved_vae')
