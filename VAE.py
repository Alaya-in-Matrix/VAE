import torch
import numpy as np
import pyro
import torch.nn               as nn
import torch.nn.functional    as F
import torchvision            as vision
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

        ndf         = 128 # number of filters
        kernel_size = 4
        stride      = 2
        padding     = 1

        # self.deconv1 = nn.ConvTranspose2d(self.z_dim, 8 * ndf, kernel_size = 8)
        self.bn1     = nn.BatchNorm2d(8 * ndf) # 1024 * 8 * 8

        self.deconv2 = nn.ConvTranspose2d(8 * ndf, 4 * ndf, kernel_size = kernel_size, stride = stride, padding = padding)
        self.bn2     = nn.BatchNorm2d(4 * ndf) # 512 * 16 * 16

        self.deconv3 = nn.ConvTranspose2d(4 * ndf, 2 * ndf, kernel_size = kernel_size, stride = stride, padding = padding)
        self.bn3     = nn.BatchNorm2d(2 * ndf) # 256 * 32 * 32

        self.deconv4 = nn.ConvTranspose2d(2 * ndf, ndf, kernel_size = kernel_size, stride = stride, padding = padding)
        self.bn4     = nn.BatchNorm2d(ndf) # 128 * 64 * 64

        self.deconv5 = nn.ConvTranspose2d(ndf, self.n_channel, kernel_size = kernel_size, stride = stride, padding = padding) # 3 * 128 * 128

    def forward(self, z):
        # h1      = F.relu(self.bn1(self.deconv1(z.view(-1, self.z_dim, 1, 1))))
        h1      = F.relu(self.bn1(F.interpolate(z.view(-1, self.z_dim, 1, 1), scale_factor = 8, mode = 'bilinear', align_corners = True)))
        h2      = F.relu(self.bn2(self.deconv2(h1)))
        h3      = F.relu(self.bn3(self.deconv3(h2)))
        h4      = F.relu(self.bn4(self.deconv4(h3)))
        h5      = self.deconv5(h4)
        rec_img = torch.tanh(h5)
        return rec_img

class Encoder(nn.Module):
    def __init__(self, n_channel, img_size, z_dim):
        super(Encoder, self).__init__()
        self.img_size  = img_size
        self.n_channel = n_channel
        self.z_dim     = z_dim

        ndf         = 128 # number of filters
        kernel_size = 4
        stride      = 2
        padding     = 1

        self.model    = vision.models.resnet18()
        self.model.fc = nn.Linear(512, self.z_dim * 2)

        # self.conv1 = nn.Conv2d(self.n_channel, ndf, kernel_size = kernel_size, stride = stride, padding= padding)

        # self.conv2 = nn.Conv2d(1 * ndf, 2 * ndf, kernel_size = kernel_size, stride = stride, padding= padding)
        # self.bn2   = nn.BatchNorm2d(2 * ndf)

        # self.conv3 = nn.Conv2d(2 * ndf, 4 * ndf, kernel_size = kernel_size, stride = stride, padding= padding)
        # self.bn3   = nn.BatchNorm2d(4 * ndf)

        # self.conv4_1 = nn.Conv2d(4 * ndf, self.z_dim, kernel_size = kernel_size, stride = stride, padding= padding)
        # self.bn4_1   = nn.BatchNorm2d(self.z_dim)

        # self.conv4_2 = nn.Conv2d(4 * ndf, self.z_dim, kernel_size = kernel_size, stride = stride, padding= padding)
        # self.bn4_2   = nn.BatchNorm2d(self.z_dim)

    def forward(self, x):
        # h1 = F.leaky_relu(self.conv1(x))
        # h2 = F.leaky_relu(self.bn2(self.conv2(h1)))
        # h3 = F.leaky_relu(self.bn3(self.conv3(h2)))

        # h_loc   = self.bn4_1(self.conv4_1(h3))
        # h_scale = self.bn4_2(self.conv4_2(h3))

        # z_loc   = F.avg_pool2d(h_loc, h_loc.shape[2]).squeeze()
        # z_scale = F.avg_pool2d(h_loc, h_scale.shape[2]).squeeze()
        features = self.model(x)
        z_loc    = features[:, :self.z_dim]
        z_scale  = features[:, self.z_dim:]

        return z_loc, torch.exp(z_scale)

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
            loss = self.svi.step(imgs)
            epoch_loss += loss
        epoch_loss /= len(loader.dataset)
        return epoch_loss

    def model(self, x):
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", len(x)):
            z_loc       = x.new_zeros(torch.Size((len(x), self.z_dim)))
            z_scale     = x.new_ones( torch.Size((len(x), self.z_dim)))
            z           = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            loc_img     = self.decoder(z)
            # pyro.sample("obs", dist.Bernoulli(0.5 * loc_img + 0.5).to_event(3), obs = 0.5 * x + 0.5)
            noise_level = 0.05
            pyro.sample("obs", dist.Normal(loc_img, noise_level).to_event(3), obs=x)

    def guide(self, x):
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            z_loc, z_scale = self.encoder.forward(x)
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    def reconstruct_img(self, x):
        with torch.no_grad():
            if self.use_cuda:
                x = x.cuda()
            z_loc, z_scale = self.encoder(x)
            z              = dist.Normal(z_loc, z_scale).sample()
            loc_img        = self.decoder(z)
        return loc_img.cpu()
