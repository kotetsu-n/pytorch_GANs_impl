'''
Refered to
https://github.com/malzantot/Pytorch-conditional-GANs/blob/master/conditional_dcgan.py
'''

import random
import math
import time
import pandas as pd
import numpy as np
from PIL import Image
import argparse
import os

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from torchvision import transforms
from torchvision import datasets
from torchvision.utils import save_image

import matplotlib.pyplot as plt

class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10, img_shape=(1,32,32)):
        super(Generator, self).__init__()

        # is it Okay to change one-hot-vector?
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.label_emb = nn.Embedding(self.num_classes, self.num_classes)
        self.img_shape = img_shape

        def block(in_dim, out_dim, normalize=True):
            layers = [nn.Linear(in_dim, out_dim)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim+self.num_classes, 128),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z, labels):
        gen_input = torch.cat((self.label_emb(labels), z), -1)
        out = self.model(gen_input)
        out = out.view(out.size(0), *self.img_shape)

        return out

class Discriminator(nn.Module):
    def __init__(self, num_classes=10, img_shape=(1,32,32)):
        super(Discriminator, self).__init__()
        
        self.num_classes = num_classes
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.img_shape = img_shape

        self.model = nn.Sequential(
            nn.Linear(self.num_classes+int(np.prod(self.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1)
        )

    def forward(self, x, labels):
        d_in = torch.cat((x.view(x.size(0), -1), self.label_emb(labels)), -1)
        out = self.model(d_in)

        return out

def download_MNIST(img_size, batch_size):
    os.makedirs("../../data/mnist", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../../data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    return dataloader

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def train_model(G, D, opt, dataloader, num_epochs):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('currently use:', device)

    # optimization
    g_optimizer = torch.optim.Adam(G.parameters(), opt.lr_g, [opt.b1, opt.b2])
    d_optimizer = torch.optim.Adam(D.parameters(), opt.lr_d, [opt.b1, opt.b2])

    # criterion = nn.BCEWithLogitsLoss(reduction='mean')
    criterion = nn.MSELoss()

    G.to(device)
    D.to(device)

    G.train()
    D.train()

    torch.backends.cudnn.benchmark = True

    os.makedirs('./images/', exist_ok=True)

    for epoch in range(num_epochs):
        t_epoch_start = time.time()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        iteration = 0

        print('----------')
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('----------')
        print(' (train) ')

        for i, (imgs, labels) in enumerate(dataloader):
            if imgs.size()[0] == 1:
                continue

            real_imgs = imgs.to(device)
            real_labels = labels.to(device)

            batch_size = imgs.size()[0]
            label_1 = torch.full((batch_size,1), 1.0).to(device)
            label_0 = torch.full((batch_size,1), 0.0).to(device)

            '''
             train generator
            '''
            g_optimizer.zero_grad()

            # generate noises for generator
            z = torch.randn(batch_size, opt.latent_dim).to(device)
            fake_labels = torch.randint(0, opt.num_classes, (batch_size,)).to(device)
            
            fake_imgs = G(z, fake_labels)
            d_out_fake = D(fake_imgs, fake_labels)

            g_loss = criterion(d_out_fake, label_1)

            g_loss.backward()
            g_optimizer.step()

            '''
             train discriminator
            '''

            d_optimizer.zero_grad()

            # loss for real images
            d_out_real = D(real_imgs, real_labels)
            d_loss_real = criterion(d_out_real, label_1)
            
            # loss for fake images
            d_out_fake = D(fake_imgs.detach(), fake_labels)
            d_loss_fake = criterion(d_out_fake, label_0)

            d_loss = (d_loss_real + d_loss_fake) * 0.5

            d_loss.backward()
            d_optimizer.step()


            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            iteration += 1
     
            # print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            #       % (epoch, opt.num_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            # )

            def sample_image(generator, iteration, opt, device, n_row=10):
                batches_done=iteration
                z = torch.randn(n_row ** 2, opt.latent_dim).to(device)
                labels = np.array([num for _ in range(n_row) for num in range(n_row)])
                labels = torch.tensor(labels).to(device)
                gen_imgs = generator(z, labels)
                save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)

            if iteration % opt.sample_interval == 0:
                sample_image(G, iteration, opt, device, n_row=10)
        
        t_epoch_finish = time.time()
        print('-------------')
        print('epoch {} || Epoch_D_Loss:{:.4f} ||Epoch_G_Loss:{:.4f}'.format(
            epoch, epoch_d_loss/iteration, epoch_g_loss/iteration))
        print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
        t_epoch_start = time.time()

    return G, D

def train(opt):
    dataloader = download_MNIST(img_size=opt.img_size, batch_size=opt.batch_size)

    batch_iterator = iter(dataloader)
    imgs = next(batch_iterator)
    # print(imgs.size())

    G = Generator(latent_dim=opt.latent_dim, num_classes=opt.num_classes, 
                  img_shape=(opt.channels,opt.img_size, opt.img_size))
    D = Discriminator(num_classes=opt.num_classes, 
                      img_shape=(opt.channels,opt.img_size, opt.img_size))

    # G.apply(weights_init)
    # D.apply(weights_init)

    num_epochs = opt.num_epochs
    G_update, D_update = train_model(
        G, D, opt, dataloader=dataloader, num_epochs=num_epochs)

    return G_update, D_update

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr_d", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--lr_g", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--num_classes", type=int, default=10, help="number of classes for dataset")
    parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
    opt = parser.parse_args(args=['--num_epochs', '200'])
    print(opt)

    train(opt)