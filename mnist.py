# from VAE import VAE
# import torch
# import torch.nn as nn
# import torchvision.datasets   as datasets
# import torchvision.transforms as transforms

# def setup_data_loaders(batch_size = 128, use_cuda = False):
#     root         = './data'
#     download     = True
#     trans        = transforms.ToTensor()
#     train_set    = datasets.MNIST(root = root, train = True,  transform = trans, download = download)
#     test_set     = datasets.MNIST(root = root, train = False, transform = trans)
#     kwargs       = {'pin_memory': use_cuda}
#     train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True,  **kwargs)
#     test_loader  = torch.utils.data.DataLoader(dataset=test_set,  batch_size=batch_size, shuffle=False, **kwargs)
#     return train_loader, test_loader

# def train(svi, train_loader, use_cuda=False):
#     epoch_loss = 0.
#     for x, _ in train_loader:
#         if use_cuda:
#             x = x.cuda()
#         loss        = svi.step(x)
#         epoch_loss += loss
#     normalizer_train       = len(train_loader.dataset)
#     total_epoch_loss_train = epoch_loss / normalizer_train
#     return total_epoch_loss_train

# def evaluate(svi, test_loader, use_cuda=False):
#     test_loss = 0.
#     for x, _ in test_loader:
#         if use_cuda:
#             x = x.cuda()
#         test_loss += svi.evaluate_loss(x)
#     normalizer_test = len(test_loader.dataset)
#     total_epoch_loss_test = test_loss / normalizer_test
#     return total_epoch_loss_test

# use_cuda    = False
# pyro.clear_param_store()
# vae         = VAE(use_cuda = use_cuda)
# optimizer   = Adam({"lr": 1e-3})
# svi         = SVI(vae.model, vae.guide, optimizer, loss = Trace_ELBO())
# train_loader, test_loader = setup_data_loaders(use_cuda=use_cuda, batch_size=128)
# print_every = 1
# num_epochs  = 10

# print(len(train_loader))
# print(len(test_loader))


# train_elbo = []
# test_elbo  = []
# for epoch in range(num_epochs):
#     total_epoch_loss_train = train(svi, train_loader, use_cuda=use_cuda)
#     train_elbo.append(-total_epoch_loss_train)
#     print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

#     if epoch % print_every == 0:
#         total_epoch_loss_test = evaluate(svi, test_loader, use_cuda=use_cuda)
#         test_elbo.append(-total_epoch_loss_test)
#         print("[epoch %03d]  average test loss: %.4f" % (epoch, total_epoch_loss_test))
