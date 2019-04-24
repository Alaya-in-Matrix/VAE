import sys
import torch
import numpy as np
import torch.nn               as nn
import torch.optim            as optim
import torch.nn.functional    as F
import torchvision            as vision
import torchvision.datasets   as datasets
import torchvision.transforms as transforms
from   torch.distributions    import Normal, kl_divergence
from   tqdm import tqdm

class Encoder(nn.Module):
    def __init__(self, n_channel, img_size, z_dim):
        super(Encoder, self).__init__()
        self.img_size  = img_size
        self.n_channel = n_channel
        self.z_dim     = z_dim

        # Directly use resnet18 as feature extractor
        self.model    = vision.models.resnet18()
        self.model.fc = nn.Linear(512, self.z_dim * 2)
        
        # ndf         = 64 # number of filters
        # kernel_size = 4
        # stride      = 2
        # padding     = 1
        # self.ndf    = ndf

        # self.conv1  = nn.Conv2d(self.n_channel, ndf, kernel_size = kernel_size, stride = stride, padding= padding, bias = False)

        # self.conv2  = nn.Conv2d(1 * ndf, 2 * ndf, kernel_size = kernel_size, stride = stride, padding= padding, bias = False)
        # self.bn2    = nn.BatchNorm2d(2 * ndf)

        # self.conv3  = nn.Conv2d(2 * ndf, 4 * ndf, kernel_size = kernel_size, stride = stride, padding= padding, bias = False)
        # self.bn3    = nn.BatchNorm2d(4 * ndf)

        # self.conv4  = nn.Conv2d(4 * ndf, 8 * ndf, kernel_size = kernel_size, stride = stride, padding= padding, bias = False)
        # self.bn4    = nn.BatchNorm2d(8 * ndf)

        # self.final_img_size = self.img_size // 16
        # self.fc             = nn.Linear(8 * ndf * self.final_img_size**2, 2 * self.z_dim)

    def forward(self, x):
        # h1 = F.leaky_relu(self.conv1(x),            negative_slope=0.2)
        # h2 = F.leaky_relu(self.bn2(self.conv2(h1)), negative_slope=0.2)
        # h3 = F.leaky_relu(self.bn3(self.conv3(h2)), negative_slope=0.2)
        # h4 = F.leaky_relu(self.bn4(self.conv4(h3)), negative_slope=0.2)

        # zfeatures = self.fc(h4.view(-1, 8 * self.ndf * self.final_img_size**2))
        # z_loc     = zfeatures[:, :self.z_dim]
        # z_scale   = zfeatures[:, self.z_dim:]

        features = self.model(x)
        z_loc    = features[:, :self.z_dim].squeeze()
        z_scale  = features[:, self.z_dim:].squeeze()

        return z_loc, torch.exp(z_scale) + 1e-6

class Decoder(nn.Module):
    def __init__(self, n_channel, img_size, z_dim):
        super(Decoder, self).__init__()
        self.img_size  = img_size
        self.n_channel = n_channel
        self.z_dim     = z_dim

        ndf         = 32 # number of filters
        kernel_size = 4
        stride      = 2
        padding     = 1
        self.ndf    = ndf

        self.init_size = self.img_size // 16

        # self.deconv1 = nn.ConvTranspose2d(self.z_dim, 8 * ndf, kernel_size = int(self.img_size / 64) * kernel_size, bias = False) #XXX
        # self.bn1       = nn.BatchNorm2d(8 * ndf)  
        self.linear1   = nn.Linear(self.z_dim, 8 * ndf * self.init_size * self.init_size)
        self.bn1       = nn.BatchNorm2d(8 * ndf)  

        self.deconv2 = nn.ConvTranspose2d(8 * ndf, 4 * ndf, kernel_size = kernel_size, stride = stride, padding = padding, bias = False)
        self.bn2     = nn.BatchNorm2d(4 * ndf)  

        self.deconv3 = nn.ConvTranspose2d(4 * ndf, 2 * ndf, kernel_size = kernel_size, stride = stride, padding = padding, bias = False)
        self.bn3     = nn.BatchNorm2d(2 * ndf)  

        self.deconv4 = nn.ConvTranspose2d(2 * ndf, ndf, kernel_size = kernel_size, stride = stride, padding = padding, bias = False)
        self.bn4     = nn.BatchNorm2d(ndf)      

        self.deconv5_1 = nn.ConvTranspose2d(ndf, self.n_channel, kernel_size = kernel_size, stride = stride, padding = padding, bias = False) 
        self.deconv5_2 = nn.ConvTranspose2d(ndf, self.n_channel, kernel_size = kernel_size, stride = stride, padding = padding, bias = False) 

    def forward(self, z):
        h1        = F.relu(self.bn1(self.linear1(z.view(-1, self.z_dim)).view(-1, 8 * self.ndf, self.init_size, self.init_size)))
        h2        = F.relu(self.bn2(self.deconv2(h1)))
        h3        = F.relu(self.bn3(self.deconv3(h2)))
        h4        = F.relu(self.bn4(self.deconv4(h3)))
        h5_1      = self.deconv5_1(h4)
        h5_2      = self.deconv5_1(h4)
        rec_img   = torch.tanh(h5_1)
        noise_var = torch.exp(h5_2)
        return rec_img, noise_var


class VAE(nn.Module):
    def __init__(self, n_channel, img_size = 64, z_dim=128, use_cuda=False, conf = dict()):
        assert(not (img_size & (img_size-1)))
        super(VAE, self).__init__()
        self.img_size = img_size
        self.use_cuda = use_cuda
        self.z_dim    = z_dim
        self.encoder  = Encoder(n_channel, img_size, z_dim)
        self.decoder  = Decoder(n_channel, img_size, z_dim)
        if use_cuda:
            self.cuda()
        self.noise_level = conf.get('noise_level',0.01)
        self.lr          = conf.get('lr', 3e-4)
    
    def forward(self, img):
        z_loc, z_scale     = self.encoder(img.view(-1, 3, self.img_size, self.img_size))
        z                  = Normal(z_loc, z_scale).rsample() # rsample, automatical reparameterization
        rec_img, noise_var = self.decoder(z)
        return z_loc, z_scale, rec_img, noise_var

    def loss(self, z_loc, z_scale, img_loc, img_var, true_img):
        # noise_level = torch.sqrt(self.noise_level**2 + img_var)
        # noise_level = 0.1
        # rec_loss    = -1 * Normal(img_loc, noise_level).log_prob(true_img)
        rec_loss    = 0.5 * torch.pow(img_loc - true_img, 2)
        kl_div      = kl_divergence(Normal(z_loc, z_scale), Normal(z_loc.new_zeros(1), z_scale.new_ones(1)))
        return rec_loss.sum(), 100. * kl_div.sum()

    def one_epoch(self, loader):
        opt            = optim.Adam(self.parameters(), lr = self.lr, weight_decay = 1e-4)
        epoch_rec_loss = 0.
        epoch_kl_div   = 0.
        tbar           = tqdm(loader)
        for batch, _ in tbar:
            opt.zero_grad()
            z_loc, z_scale, rec_img, noise_var = self.forward(batch)
            rec_loss, kl_div  = self.loss(z_loc, z_scale, rec_img, noise_var, batch)
            loss              = rec_loss + kl_div
            epoch_rec_loss   += rec_loss / len(loader.dataset)
            epoch_kl_div     += kl_div   / len(loader.dataset)
            loss.backward()
            opt.step()
            tbar.set_description('Train %10.2f %10.2f' % (epoch_rec_loss, epoch_kl_div))
        return epoch_rec_loss, epoch_kl_div

    def evaluate(self, loader):
        epoch_rec_loss = 0.
        epoch_kl_div   = 0.
        with torch.no_grad():
            tbar = tqdm(loader)
            for batch, _ in tbar:
                z_loc, z_scale, rec_img, noise_var  = self.forward(batch)
                rec_loss, kl_div  = self.loss(z_loc, z_scale, rec_img, noise_var, batch)
                loss              = rec_loss + kl_div
                epoch_rec_loss   += rec_loss / len(loader.dataset)
                epoch_kl_div     += kl_div   / len(loader.dataset)
                tbar.set_description('Test %10.2f %10.2f' % (epoch_rec_loss, epoch_kl_div))
        return epoch_rec_loss, epoch_kl_div

    def random_sample(self, num_samples = 8):
        z = Normal(torch.tensor(0.), torch.tensor(1.)).sample((num_samples, self.z_dim))
        if self.use_cuda:
            z = z.cuda()
        rec_img, _ = self.decoder(z)
        return rec_img.cpu()

    def reconstruct_img(self, imgs):
        if self.use_cuda:
            imgs = imgs.cuda()
        _, _, rec_img, _ = self.forward(imgs)
        return rec_img.cpu()
