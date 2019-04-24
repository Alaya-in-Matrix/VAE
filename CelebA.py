from VAE import VAE,Encoder,Decoder
import torch
import torch.nn as nn
import pyro
import pyro.optim
import torchvision as vision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid,save_image
import numpy as np
from   tqdm import tqdm,trange

# Trained with 4000 hentai images from https://github.com/alexkimxyz/nsfw_data_scraper

img_size    = 64
batch_size  = 64
use_cuda    = True
num_epochs  = 25
z_dim       = 512
lr          = 3e-4
noise_level = 1e-3
transf      = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((img_size,img_size)), 
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

images                    = datasets.ImageFolder('./data/CelebA/', transform=transf)
len_train                 = int(0.95 * len(images))
len_validate              = len(images) - len_train
train_imgs, validate_imgs = torch.utils.data.random_split(images, [len_train, len_validate])
train_loader              = torch.utils.data.DataLoader(dataset = train_imgs,    pin_memory=use_cuda, shuffle=True, batch_size=batch_size)
validate_loader           = torch.utils.data.DataLoader(dataset = validate_imgs, pin_memory=use_cuda, shuffle=True, batch_size=batch_size)
conf                      = dict()
conf['noise_level']       = noise_level
conf['lr']                = lr
vae                       = VAE(n_channel=3,img_size=img_size,z_dim = z_dim, use_cuda = use_cuda, conf = conf)
tbar = tqdm(range(num_epochs))
fid = open('losses','w')
for epoch in tbar:
    train_loss    = vae.one_epoch(train_loader)
    validate_loss = vae.evaluate(validate_loader)
    #print('Epoch %3d, train_loss = %11.2f valid_loss = %11.2f' % (epoch, train_loss, validate_loss), flush = True)
    if (epoch + 1) % 100 == 0:
        torch.save(vae, 'saved_vae')
    vae.eval()
    bx_train,_   = iter(train_loader).next()
    bx_valid,_   = iter(validate_loader).next()
    zloc_train,_ = vae.encoder(bx_train.cuda())
    zloc_valid,_ = vae.encoder(bx_valid.cuda())
    z            = torch.zeros(3*batch_size,z_dim).cuda()
    z[:batch_size,:]              = pyro.distributions.Normal(0., 1.).sample((batch_size,z_dim)).cuda()
    z[batch_size:2*batch_size,:]  = zloc_train
    z[2*batch_size:,:]            = zloc_valid

    dec_imgs = 0.5 + 0.5 * vae.decoder(z)[0].cpu()
    imgs     = make_grid(torch.cat((dec_imgs,0.5 + 0.5*bx_train,0.5 + 0.5*bx_valid),dim=0))
    save_image(imgs,'img_%d.png' % epoch)
    vae.train()
    tbar.set_description('%10.3f %10.3f' % (train_loss, validate_loss))
    fid.write('%10.3f %10.3f\n' % (train_loss, validate_loss))
    fid.flush()
fid.close()
vae.eval()
torch.save(vae, 'saved_vae')
