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

img_size    = 64
batch_size  = 64
use_cuda    = True
num_epochs  = 50
z_dim       = 512
lr          = 3e-4
noise_level = 1e-3
kl_factor   = 1.

mean      = [0.485, 0.456, 0.406] # imagenet normalization
std       = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean, std)
transf    = transforms.Compose([
    transforms.Resize((img_size,img_size)), 
    transforms.ToTensor(),
    normalize
])
images = datasets.ImageFolder('./data/CelebA/', transform=transf)

len_train                 = int(0.95 * len(images))
len_validate              = len(images) - len_train
train_imgs, validate_imgs = torch.utils.data.random_split(images, [len_train, len_validate])
train_loader              = torch.utils.data.DataLoader(dataset = train_imgs,    pin_memory=use_cuda, shuffle=True, batch_size=batch_size)
validate_loader           = torch.utils.data.DataLoader(dataset = validate_imgs, pin_memory=use_cuda, shuffle=True, batch_size=batch_size)
conf                      = dict()
conf['noise_level']       = noise_level
conf['lr']                = lr
conf['kl_factor']         = kl_factor
conf['norm_mean']         = mean
conf['norm_std']          = std
vae                       = VAE(n_channel=3,img_size=img_size,z_dim = z_dim, use_cuda = use_cuda, conf = conf)
tbar                      = tqdm(range(num_epochs))
fid                       = open('losses', 'w')
for epoch in tbar:
    train_rec, train_kl = vae.one_epoch(train_loader)
    valid_rec, valid_kl = vae.evaluate(validate_loader)

    vae.eval()
    bx_train,_ = iter(train_loader).next()
    bx_valid,_ = iter(validate_loader).next()
    bx_train   = bx_train[:8]
    bx_valid   = bx_valid[:8]
    rand_samp  = vae.random_sample(num_samples = 8)
    rec_train  = vae.reconstruct_img(bx_train)
    rec_valid  = vae.reconstruct_img(bx_valid)
    bx_train   = vae.unnormalize(bx_train)
    bx_valid   = vae.unnormalize(bx_valid)
    show_imgs  = make_grid(torch.cat((rand_samp, bx_train, rec_train, bx_valid, rec_valid), dim = 0), nrow=8)
    save_image(show_imgs,'img_%d.png' % epoch)
    vae.train()

    tbar.set_description('%8.3f %8.3f %8.3f %8.3f' % (train_rec ,  train_kl , valid_rec, valid_kl))
    fid.write('%8.3f %8.3f %8.3f %8.3f\n' % (train_rec ,  train_kl , valid_rec, valid_kl))
    fid.flush()

fid.close()

vae.eval()
torch.save(vae, 'saved_vae')
