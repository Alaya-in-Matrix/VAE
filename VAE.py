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
from   tqdm import tqdm

class Decoder(nn.Module):
    def __init__(self, n_channel, img_size, z_dim):
        super(Decoder, self).__init__()
        assert(img_size == 64 or img_size == 128)
        self.img_size  = img_size
        self.n_channel = n_channel
        self.z_dim     = z_dim

        ndf         = 64 # number of filters
        kernel_size = 4
        stride      = 2
        padding     = 1

        self.deconv1 = nn.ConvTranspose2d(self.z_dim, 8 * ndf, kernel_size = int(self.img_size / 64) * kernel_size, bias = False) #XXX
        self.bn1     = nn.BatchNorm2d(8 * ndf) # 1024 * 8 * 8

        self.deconv2 = nn.ConvTranspose2d(8 * ndf, 4 * ndf, kernel_size = kernel_size, stride = stride, padding = padding, bias = False)
        self.bn2     = nn.BatchNorm2d(4 * ndf) # 512 * 16 * 16

        self.deconv3 = nn.ConvTranspose2d(4 * ndf, 2 * ndf, kernel_size = kernel_size, stride = stride, padding = padding, bias = False)
        self.bn3     = nn.BatchNorm2d(2 * ndf) # 256 * 32 * 32

        self.deconv4 = nn.ConvTranspose2d(2 * ndf, ndf, kernel_size = kernel_size, stride = stride, padding = padding, bias = False)
        self.bn4     = nn.BatchNorm2d(ndf) # 128 * 64 * 64

        self.deconv5_1 = nn.ConvTranspose2d(ndf, self.n_channel, kernel_size = kernel_size, stride = stride, padding = padding, bias = False) # 3 * 128 * 128
        self.deconv5_2 = nn.ConvTranspose2d(ndf, self.n_channel, kernel_size = kernel_size, stride = stride, padding = padding, bias = False) # 3 * 128 * 128

    def forward(self, z):
        h1          = F.relu(self.bn1(self.deconv1(z.view(-1, self.z_dim, 1, 1))))
        h2          = F.relu(self.bn2(self.deconv2(h1)))
        h3          = F.relu(self.bn3(self.deconv3(h2)))
        h4          = F.relu(self.bn4(self.deconv4(h3)))
        h5_1        = self.deconv5_1(h4)
        h5_2        = self.deconv5_1(h4)
        rec_img     = torch.tanh(h5_1)
        noise_level = torch.exp(h5_2)
        return rec_img, noise_level

class Encoder(nn.Module):
    def __init__(self, n_channel, img_size, z_dim):
        super(Encoder, self).__init__()
        self.img_size  = img_size
        self.n_channel = n_channel
        self.z_dim     = z_dim


        self.model    = vision.models.resnet18()
        self.model.fc = nn.Linear(512, self.z_dim * 2)
        

        # ndf          = 64 # number of filters
        # kernel_size  = 4
        # stride       = 2
        # padding      = 1

        # self.conv1   = nn.Conv2d(self.n_channel, ndf, kernel_size = kernel_size, stride = stride, padding= padding, bias = False)

        # self.conv2   = nn.Conv2d(1 * ndf, 2 * ndf, kernel_size = kernel_size, stride = stride, padding= padding, bias = False)
        # self.bn2     = nn.BatchNorm2d(2 * ndf)

        # self.conv3   = nn.Conv2d(2 * ndf, 4 * ndf, kernel_size = kernel_size, stride = stride, padding= padding, bias = False)
        # self.bn3     = nn.BatchNorm2d(4 * ndf)

        # self.conv4_1 = nn.Conv2d(4 * ndf, self.z_dim, kernel_size = kernel_size, stride = stride, padding= padding, bias = False)
        # self.bn4_1   = nn.BatchNorm2d(self.z_dim)

        # self.conv4_2 = nn.Conv2d(4 * ndf, self.z_dim, kernel_size = kernel_size, stride = stride, padding= padding, bias = False)
        # self.bn4_2   = nn.BatchNorm2d(self.z_dim)
        # 
        # self.zloc_linear   = nn.Linear(self.z_dim * 4 * 4, self.z_dim)
        # self.zscale_linear = nn.Linear(self.z_dim * 4 * 4, self.z_dim)

    def forward(self, x):
        # h1 = F.leaky_relu(self.conv1(x))
        # h2 = F.leaky_relu(self.bn2(self.conv2(h1)))
        # h3 = F.leaky_relu(self.bn3(self.conv3(h2)))

        # h_loc   = self.bn4_1(self.conv4_1(h3))
        # h_scale = self.bn4_2(self.conv4_2(h3))

        # z_loc   = self.zloc_linear(h_loc.view(  -1, self.z_dim * 4 * 4)).squeeze()
        # z_scale = self.zscale_linear(h_loc.view(-1, self.z_dim * 4 * 4)).squeeze()

        features = self.model(x)
        z_loc    = features[:, :self.z_dim].squeeze()
        z_scale  = features[:, self.z_dim:].squeeze()

        return z_loc, F.softplus(z_scale) + 1e-6

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
        self.noise_level = conf.get('noise_level',0.1)
        self.lr          = conf.get('lr', 1e-4)
        self.optim       = Adam({"lr": self.lr})
        self.svi         = pyro.infer.SVI(self.model, self.guide, self.optim, loss = Trace_ELBO())

    def one_epoch(self, loader):
        epoch_loss = 0.
        for imgs, _ in tqdm(loader,desc='Training'):
            if self.use_cuda:
                imgs = imgs.cuda()
            loss = self.svi.step(imgs)
            #print('%11.2f' % (loss / len(imgs)),flush = True)
            epoch_loss += loss
        epoch_loss /= len(loader.dataset)
        return epoch_loss

    def evaluate(self, loader):
        epoch_loss = 0.
        for imgs, _ in tqdm(loader,desc='Testing '):
            if self.use_cuda:
                imgs = imgs.cuda()
            loss = self.svi.evaluate_loss(imgs)
            # print('%11.2f' % (loss / len(imgs)))
            epoch_loss += loss
        epoch_loss /= len(loader.dataset)
        return epoch_loss

    def model(self, x):
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", len(x)):
            z_loc                = x.new_zeros(torch.Size((len(x), self.z_dim)))
            z_scale              = x.new_ones( torch.Size((len(x), self.z_dim)))
            z                    = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            loc_img, noise_level = self.decoder(z)
            # pyro.sample("obs", dist.Bernoulli(0.5 * loc_img + 0.5).to_event(3), obs = 0.5 * x + 0.5)
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
            loc_img, _     = self.decoder(z)
        return loc_img.cpu()
