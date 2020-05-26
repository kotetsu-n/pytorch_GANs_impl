'''
refer to 
https://github.com/YutaroOgawa/pytorch_advanced/blob/master/5_gan_generation/5-1-2_DCGAN.ipynb
'''

import os
import random
import math
import time
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms

import matplotlib.pyplot as plt

import sklearn
from sklearn.datasets import fetch_openml

def data_preparation_for_dcgan():
    data_dir = "./data/"
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    mnist = fetch_openml('mnist_784', version=1, data_home="./data/")  

    def test_loading_mnist(mnist, visualize=False):
        X = mnist.data
        y = mnist.target

        if visualize:
            plt.imshow(X[0].reshape(28, 28), cmap='gray')
        return X, y

    X, y = test_loading_mnist(mnist)

    data_dir_path = "./data/img_78/"
    if not os.path.exists(data_dir_path):
        os.mkdir(data_dir_path)

    count7=0
    count8=0
    max_num=200

    for i in range(len(X)):
        if (y[i] is "7") and (count7<max_num):
            file_path="./data/img_78/img_7_"+str(count7)+".jpg"
            im_f=(X[i].reshape(28, 28)) 
            pil_img_f = Image.fromarray(im_f.astype(np.uint8))
            pil_img_f = pil_img_f.resize((64, 64), Image.BICUBIC)
            pil_img_f.save(file_path)
            count7+=1 
        
        if (y[i] is "8") and (count8<max_num):
            file_path="./data/img_78/img_8_"+str(count8)+".jpg"
            im_f=(X[i].reshape(28, 28))
            pil_img_f = Image.fromarray(im_f.astype(np.uint8))
            pil_img_f = pil_img_f.resize((64, 64), Image.BICUBIC)
            pil_img_f.save(file_path)
            count8+=1

class Generator(nn.Module):
    def __init__(self, z_dim=20, image_size=64):
        super(Generator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(z_dim, image_size * 8,
                               kernel_size=4, stride=1),
            nn.BatchNorm2d(image_size * 8),
            nn.ReLU(inplace=True))
        
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(image_size*8, image_size*4,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(image_size*4),
            nn.ReLU(inplace=True))
        
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(image_size*4, image_size*2,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(image_size*2),
            nn.ReLU(inplace=True))

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(image_size*2, image_size,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(image_size),
            nn.ReLU(inplace=True))

        self.last = nn.Sequential(
            nn.ConvTranspose2d(image_size, 1,
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh())

    def forward(self, z):
        out = self.layer1(z)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.last(out)

        return out

class Discriminator(nn.Module):
    def __init__(self, z_dim=20, image_size=64):
        super(Discriminator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, image_size, 
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(image_size, image_size*2, 
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True))

        self.layer3 = nn.Sequential(
            nn.Conv2d(image_size*2, image_size*4, 
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True))

        self.layer4 = nn.Sequential(
            nn.Conv2d(image_size*4, image_size*8, 
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True))

        self.last = nn.Conv2d(image_size*8, 1, kernel_size=4, stride=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.last(out)

        return out

def make_datapath_list():
    train_img_list = list()

    for img_idx in range(200):
        img_path = "./data/img_78/img_7_" + str(img_idx)+'.jpg'
        train_img_list.append(img_path)

        img_path = "./data/img_78/img_8_" + str(img_idx)+'.jpg'
        train_img_list.append(img_path)

    return train_img_list

class ImageTransform():
    def __init__(self, mean, std):
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, img):
        return self.data_transform(img)

class GAN_Img_Dataset(data.Dataset):
    def __init__(self, file_list, transform):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path)
        img_transformed = self.transform(img)
        return img_transformed

def check_generator():
    G = Generator(z_dim=20, image_size=64)
    input_z = torch.randn(1, 20)
    input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
    fake_images = G(input_z)

    img_transformed = fake_images[0][0].detach().numpy()
    plt.imshow(img_transformed, 'gray')
    plt.show()

def check_discriminator():
    G = Generator(z_dim=20, image_size=64)
    D = Discriminator(z_dim=20, image_size=64)
    input_z = torch.randn(1, 20)
    input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
    fake_images = G(input_z)

    d_out = D(fake_images)
    print(nn.Sigmoid()(d_out))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def test_trained_result(model, iteration):
    train_img_list = make_datapath_list()

    mean = (0.5,)
    std = (0.5,)
    train_dataset = GAN_Img_Dataset(
        file_list=train_img_list, transform=ImageTransform(mean, std))
    
    batch_size = 64
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batch_size = 8
    z_dim = 20
    fixed_z = torch.randn(batch_size, z_dim)
    fixed_z = fixed_z.view(fixed_z.size(0), fixed_z.size(1), 1, 1)

    fake_images = model(fixed_z.to(device))

    batch_iterator = iter(train_dataloader)
    imges = next(batch_iterator)

    fig = plt.figure(figsize=(15, 6))
    for i in range(0, 5):
        plt.subplot(2, 5, i+1)
        plt.imshow(imges[i][0].cpu().detach().numpy(), 'gray')

        plt.subplot(2, 5, 5+i+1)
        plt.imshow(fake_images[i][0].cpu().detach().numpy(), 'gray')
    plt.savefig("images/{}.png".format(iteration))

def train_model(G, D, dataloader, num_epochs):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('currently use:', device)

    # optimization
    g_lr, d_lr = 0.0001, 0.0004
    beta1, beta2 = 0.0, 0.9
    g_optimizer = torch.optim.Adam(G.parameters(), g_lr, [beta1, beta2])
    d_optimizer = torch.optim.Adam(D.parameters(), d_lr, [beta1, beta2])

    criterion = nn.BCEWithLogitsLoss(reduction='mean')

    z_dim = 20
    mini_batch_size = 64

    G.to(device)
    D.to(device)

    G.train()
    D.train()

    torch.backends.cudnn.benchmark = True

    num_train_imgs = len(dataloader.dataset)
    batch_size = dataloader.batch_size

    iteration = 1
    logs = []

    os.makedirs('./images/', exist_ok=True)
    sample_interval = 200

    for epoch in range(num_epochs):
        t_epoch_start = time.time()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        print('----------')
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('----------')
        print(' (train) ')

        for imgs in dataloader:
            if imgs.size()[0] == 1:
                continue

            imgs = imgs.to(device)

            mini_batch_size = imgs.size()[0]
            label_real = torch.full((mini_batch_size,), 1).to(device)
            label_fake = torch.full((mini_batch_size,), 0).to(device)

            d_out_real = D(imgs)

            input_z = torch.randn(mini_batch_size, z_dim).to(device)
            input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
            fake_imgs = G(input_z)
            d_out_fake = D(fake_imgs)

            # train discriminator
            d_loss_real = criterion(d_out_real.view(-1), label_real)
            d_loss_fake = criterion(d_out_fake.view(-1), label_fake)
            d_loss = d_loss_real + d_loss_fake

            g_optimizer.zero_grad()
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # train generator
            input_z = torch.randn(mini_batch_size, z_dim).to(device)
            input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
            fake_images = G(input_z)
            d_out_fake = D(fake_images)
            g_loss = criterion(d_out_fake.view(-1), label_real)

            g_optimizer.zero_grad()
            d_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()

            iteration += 1

            if iteration % sample_interval == 0:
              test_trained_result(G, iteration)
        
        t_epoch_finish = time.time()
        print('-------------')
        print('epoch {} || Epoch_D_Loss:{:.4f} ||Epoch_G_Loss:{:.4f}'.format(
            epoch, epoch_d_loss/batch_size, epoch_g_loss/batch_size))
        print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
        t_epoch_start = time.time()

    return G, D

def train():
    train_img_list = make_datapath_list()

    mean = (0.5,)
    std = (0.5,)
    train_dataset = GAN_Img_Dataset(
        file_list=train_img_list, transform=ImageTransform(mean, std))
    
    batch_size = 64
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    batch_iterator = iter(train_dataloader)
    imgs = next(batch_iterator)
    print(imgs.size())

    G = Generator(z_dim=20, image_size=64)
    D = Discriminator(z_dim=20, image_size=64)

    G.apply(weights_init)
    D.apply(weights_init)

    num_epochs = 200
    G_update, D_update = train_model(
        G, D, dataloader=train_dataloader, num_epochs=num_epochs)

    return G_update, D_update

if __name__=='__main__':
    data_preparation_for_dcgan()
    train()