import torch
import numpy as np
import pyro
import torch.nn               as nn
import torch.nn.functional    as F
import torchvision.datasets   as datasets
import torchvision.transforms as transforms
import pyro.distributions     as dist
from   pyro.infer import SVI, Trace_ELBO
from   pyro.optim import Adam

class Decoder(nn.Module):
    def __init__(self, n_channel, img_size, z_dim):
        super(Decoder, self).__init__()
        self.img_size  = img_size
        self.n_channel = n_channel
        self.z_dim     = z_dim
        hidden_dim     = 400
        self.fc1       = nn.Linear(z_dim, hidden_dim)
        self.fc21      = nn.Linear(hidden_dim, self.n_channel * self.img_size * self.img_size)
        self.relu      = nn.ReLU()
        self.sigmoid   = nn.Sigmoid()

    def forward(self, z):
        hidden  = self.relu(self.fc1(z.view(-1, self.z_dim)))
        loc_img = self.sigmoid(self.fc21(hidden)).view(-1, self.n_channel, self.img_size, self.img_size)
        return loc_img

class Encoder(nn.Module):
    def __init__(self, n_channel, img_size, z_dim):
        super(Encoder, self).__init__()
        self.img_size  = img_size
        self.n_channel = n_channel
        self.z_dim     = z_dim
        hidden_dim     = 400
        self.fc1       = nn.Linear(self.n_channel * self.img_size**2, hidden_dim)
        self.fc21      = nn.Linear(hidden_dim, z_dim)
        self.fc22      = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        x       = self.fc1(x.view(-1, self.n_channel * self.img_size**2))
        hidden  = F.relu(x)
        z_loc   = self.fc21(hidden)
        z_scale = F.softplus(self.fc22(hidden))
        return z_loc, z_scale

class VAE(nn.Module):
    def __init__(self, n_channel, img_size = 28, z_dim=50, use_cuda=False, conf = dict()):
        super(VAE, self).__init__()
        self.img_size = img_size
        self.use_cuda = use_cuda
        self.z_dim    = z_dim
        self.encoder  = Encoder(n_channel, img_size, z_dim)
        self.decoder  = Decoder(n_channel, img_size, z_dim)
        if use_cuda:
            self.cuda()
        self.lr       = conf.get('lr', 1e-4)
        self.optim    = Adam({"lr": self.lr})
        self.svi      = pyro.infer.SVI(self.model, self.guide, self.optim, loss = Trace_ELBO())

    def one_epoch(self, loader):
        epoch_loss = 0.
        for imgs, _ in loader:
            if self.use_cuda:
                imgs = imgs.cuda()
            loss = self.svi.step(imgs) / len(imgs)
            epoch_loss += loss
        return epoch_loss

    def model(self, x):
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", len(x)):
            z_loc       = x.new_zeros(torch.Size((len(x), self.z_dim)))
            z_scale     = x.new_ones( torch.Size((len(x), self.z_dim)))
            z           = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            loc_img     = self.decoder(z)
            noise_level = 0.1 * x.new_ones(1)
            pyro.sample("obs", dist.Normal(loc_img, noise_level).to_event(3), obs=x)

    def guide(self, x):
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            z_loc, z_scale = self.encoder.forward(x)
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    def reconstruct_img(self, x):
        z_loc, z_scale = self.encoder(x)
        z              = dist.Normal(z_loc, z_scale).sample()
        loc_img        = self.decoder(z)
        return loc_img
