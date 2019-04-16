import torch
import numpy as np
import pyro
import torch.nn               as nn
import torchvision.datasets   as datasets
import torchvision.transforms as transforms
import pyro.distributions     as dist
from   pyro.infer import SVI, Trace_ELBO
from   pyro.optim import Adam


def setup_data_loaders(batch_size = 128, use_cuda = False):
    root         = './data'
    download     = True
    trans        = transforms.ToTensor()
    train_set    = datasets.MNIST(root = root, train = True,  transform = trans, download = download)
    test_set     = datasets.MNIST(root = root, train = False, transform = trans)
    kwargs       = {'pin_memory': use_cuda}
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True,  **kwargs)
    test_loader  = torch.utils.data.DataLoader(dataset=test_set,  batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader


class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.fc1     = nn.Linear(z_dim, hidden_dim)
        self.fc21    = nn.Linear(hidden_dim, 784)
        self.relu    = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        hidden  = self.relu(self.fc1(z))
        loc_img = self.sigmoid(self.fc21(hidden)) # [0, 1]
        return loc_img

class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.fc1      = nn.Linear(784, hidden_dim)
        self.fc21     = nn.Linear(hidden_dim, z_dim)
        self.fc22     = nn.Linear(hidden_dim, z_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x       = x.reshape(-1, 784)
        hidden  = self.relu(self.fc1(x))
        z_loc   = self.fc21(hidden)
        z_scale = torch.exp(self.fc22(hidden)) # make sure z_scale is positive
        return z_loc, z_scale

class VAE(nn.Module):
    def __init__(self, z_dim=50, hidden_dim=400, use_cuda=False):
        super(VAE, self).__init__()
        self.encoder = Encoder(z_dim, hidden_dim)
        self.decoder = Decoder(z_dim, hidden_dim)
        if use_cuda:
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim    = z_dim

    def model(self, x):
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            z_loc   = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
            z       = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            loc_img = self.decoder.forward(z)
            pyro.sample("obs", dist.Normal(loc_img, torch.tensor(0.1)).to_event(1), obs=x.reshape(-1, 784))
            # pyro.sample("obs", dist.Bernoulli(loc_img).to_event(1), obs=x.reshape(-1, 784))

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


def train(svi, train_loader, use_cuda=False):
    epoch_loss = 0.
    for x, _ in train_loader:
        if use_cuda:
            x = x.cuda()
        loss        = svi.step(x)
        epoch_loss += loss
    normalizer_train       = len(train_loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train
    return total_epoch_loss_train

def evaluate(svi, test_loader, use_cuda=False):
    test_loss = 0.
    for x, _ in test_loader:
        if use_cuda:
            x = x.cuda()
        test_loss += svi.evaluate_loss(x)
    normalizer_test = len(test_loader.dataset)
    total_epoch_loss_test = test_loss / normalizer_test
    return total_epoch_loss_test

use_cuda    = False
pyro.clear_param_store()
vae         = VAE(use_cuda = use_cuda)
optimizer   = Adam({"lr": 1e-3})
svi         = SVI(vae.model, vae.guide, optimizer, loss = Trace_ELBO())
train_loader, test_loader = setup_data_loaders(use_cuda=use_cuda, batch_size=128)
print_every = 1
num_epochs  = 10

print(len(train_loader))
print(len(test_loader))


train_elbo = []
test_elbo  = []
for epoch in range(num_epochs):
    total_epoch_loss_train = train(svi, train_loader, use_cuda=use_cuda)
    train_elbo.append(-total_epoch_loss_train)
    print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

    if epoch % print_every == 0:
        total_epoch_loss_test = evaluate(svi, test_loader, use_cuda=use_cuda)
        test_elbo.append(-total_epoch_loss_test)
        print("[epoch %03d]  average test loss: %.4f" % (epoch, total_epoch_loss_test))
