'''
Refered to the followings to implement
https://github.com/y-kamiya/machine-learning-samples/blob/master/python3/deep/pytorch/pix2pix.py
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
if there are any problem to go public this code, please contact me
'''

import sys
import argparse
import os.path
import os
import random
import time
import numpy as np
import glob
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image

class AlignedDataset(Dataset):
    IMG_EXTENSIONS = ['.png', 'jpg']

    def __init__(self, config):
        self.config = config
        self.imgdir = config.imgdir
        self.labeldir = config.labeldir
        self.img_paths = sorted(glob.glob(self.imgdir+'/*.jpg'))
        print(self.imgdir+'/*.jpg')
        self.label_paths = sorted(glob.glob(self.labeldir+'/*.png'))
        self.direction = config.direction

    @classmethod
    def is_image_file(self, fname):
        return any(fname.endswith(ext) for ext in self.IMG_EXTENSIONS)

    @classmethod
    def __make_dataset(self, dir):
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if self.is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
        return images

    def __transform(self, param):
        list = []

        load_size = self.config.load_size
        list.append(transforms.Resize([load_size, load_size], Image.BICUBIC))

        (x, y) = param['crop_pos']
        crop_size = self.config.crop_size
        list.append(transforms.Lambda(lambda img: img.crop((x, y, x + crop_size, y + crop_size))))

        if param['flip']:
            list.append(transforms.Lambda(lambda img: img.transpose(Image.FLIP_LEFT_RIGHT)))

        list += [transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        return transforms.Compose(list)

    def __transform_param(self):
        x_max = self.config.load_size - self.config.crop_size
        x = random.randint(0, np.maximum(0, x_max))
        y = random.randint(0, np.maximum(0, x_max))

        flip = random.random() > 0.5

        return {'crop_pos': (x, y), 'flip': flip}

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        label_path = self.label_paths[index]
        img = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('RGB')

        param = self.__transform_param()
        transform = self.__transform(param)
        if self.direction == 'A2B':
          A = transform(img)
          B = transform(label)
          A_path = img_path
          B_path = label_path
        else:
          A = transform(label)
          B = transform(img)
          A_path = label_path
          B_path = img_path

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return len(self.img_paths)

"""# Generator"""

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.down0 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)

        self.down1 = self.__down(64, 128)
        self.down2 = self.__down(128, 256)
        self.down3 = self.__down(256, 512)
        self.down4 = self.__down(512, 512)
        self.down5 = self.__down(512, 512)
        self.down6 = self.__down(512, 512)
        self.down7 = self.__down(512, 512, use_norm=False)

        self.up7 = self.__up(512, 512)
        self.up6 = self.__up(1024, 512, use_dropout=True)
        self.up5 = self.__up(1024, 512, use_dropout=True)
        self.up4 = self.__up(1024, 512, use_dropout=True)
        self.up3 = self.__up(1024, 256)
        self.up2 = self.__up(512, 128)
        self.up1 = self.__up(256, 64)

        self.up0 = nn.Sequential(
            self.__up(128, 3, use_norm=False),
            nn.Tanh(),
        )

    def __down(self, input, output, use_norm=True):
        layer = [
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(input, output, kernel_size=4, stride=2, padding=1),
        ]
        if use_norm:
            layer.append(nn.BatchNorm2d(output))

        return nn.Sequential(*layer)

    def __up(self, input, output, use_norm=True, use_dropout=False):
        layer = [
            nn.ReLU(True),
            nn.ConvTranspose2d(input, output, kernel_size=4, stride=2, padding=1),
        ]
        if use_norm:
            layer.append(nn.BatchNorm2d(output))

        if use_dropout:
            layer.append(nn.Dropout(0.5))

        return nn.Sequential(*layer)

    def forward(self, x):
        x0 = self.down0(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        x7 = self.down7(x6)

        y7 = self.up7(x7)
        y6 = self.up6(self.concat(x6, y7))
        y5 = self.up5(self.concat(x5, y6))
        y4 = self.up4(self.concat(x4, y5))
        y3 = self.up3(self.concat(x3, y4))
        y2 = self.up2(self.concat(x2, y3))
        y1 = self.up1(self.concat(x1, y2))
        y0 = self.up0(self.concat(x0, y1))

        return y0

    def concat(self, x, y):
        return torch.cat([x, y], dim=1)

"""# Discriminator"""

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            self.__layer(64, 128),
            self.__layer(128, 256),
            self.__layer(256, 512, stride=1),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
        )

    def __layer(self, input, output, stride=2):
        return nn.Sequential(
            nn.Conv2d(input, output, kernel_size=4, stride=stride, padding=1),
            nn.BatchNorm2d(output),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x):
        return self.model(x)

"""# Loss"""

class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()

        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))
        self.loss = nn.BCEWithLogitsLoss()

    def __call__(self, prediction, is_real):
        if is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label

        return self.loss(prediction, target_tensor.expand_as(prediction))

"""# Trainer"""

class Pix2Pix():
    def __init__(self, config):
        self.config = config
        self.netG = Generator().to(self.config.device)
        self.netG.apply(self.__weights_init)
        if self.config.generator != None:
            self.netG.load_state_dict(torch.load(self.config.generator, map_location=self.config.device_name), strict=False)

        self.netD = Discriminator().to(self.config.device)
        self.netD.apply(self.__weights_init)
        if self.config.discriminator != None:
            self.netD.load_state_dict(torch.load(self.config.discriminator, map_location=self.config.device_name), strict=False)

        self.optimizerG = optim.Adam(self.netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.criterionGAN = GANLoss().to(self.config.device)
        self.criterionL1 = nn.L1Loss()

        self.schedulerG = optim.lr_scheduler.LambdaLR(self.optimizerG, self.__modify_learning_rate)
        self.schedulerD = optim.lr_scheduler.LambdaLR(self.optimizerD, self.__modify_learning_rate)

        self.training_start_time = time.time()
        self.append_log(config)
        self.append_log(self.netG)
        self.append_log(self.netD)

        # for log
        self.epoch_lossG, self.epoch_lossG_GAN, self.epoch_lossG_L1 = 0.0, 0.0, 0.0,
        self.epoch_lossD, self.epoch_lossD_fake, self.epoch_lossD_real = 0.0, 0.0, 0.0

    def update_learning_rate(self):
        self.schedulerG.step()
        self.schedulerD.step()

    def __modify_learning_rate(self, epoch):
        if self.config.epochs_lr_decay_start < 0:
            return 1.0

        delta = max(0, epoch - self.config.epochs_lr_decay_start) / float(self.config.epochs_lr_decay + 1)
        return max(0.0, 1.0 - delta)

    def __weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def train(self, data):
        self.realA = data['A'].to(self.config.device)
        self.realB = data['B'].to(self.config.device)

        self.fakeB = self.netG(self.realA)

        # Discriminator
        self.set_requires_grad(self.netD, True)
        self.optimizerD.zero_grad()

        fakeAB = torch.cat((self.realA, self.fakeB), dim=1)
        pred_fake = self.netD(fakeAB.detach())
        self.lossD_fake = self.criterionGAN(pred_fake, False)

        realAB = torch.cat((self.realA, self.realB), dim=1)
        pred_real = self.netD(realAB)
        self.lossD_real = self.criterionGAN(pred_real, True)

        self.lossD = (self.lossD_fake + self.lossD_real) * 0.5

        self.lossD.backward()
        self.optimizerD.step()

        # Generator
        self.set_requires_grad(self.netD, False)
        self.optimizerG.zero_grad()
        # with torch.no_grad():
        #     pred_fake = self.netD(fakeAB)
        fakeAB = torch.cat((self.realA, self.fakeB), dim=1)
        pred_fake = self.netD(fakeAB)
        self.lossG_GAN = self.criterionGAN(pred_fake, True)
        
        self.lossG_L1 = self.criterionL1(self.fakeB, self.realB) * self.config.lambda_L1

        self.lossG = self.lossG_GAN + self.lossG_L1

        self.lossG.backward()
        self.optimizerG.step()

        # for log
        # self.last_fakeB = self.fakeB
        self.epoch_lossG += self.lossG
        self.epoch_lossG_GAN += self.lossG_GAN
        self.epoch_lossG_L1 += self.lossG_L1
        self.epoch_lossD += self.lossD
        self.epoch_lossD_real += self.lossD_real
        self.epoch_lossD_fake += self.lossD_fake

    def save(self, epoch):
        output_dir = self.config.output_dir
        torch.save(self.netG.state_dict(), '{}/pix2pix_G_epoch_{}'.format(output_dir, epoch))
        torch.save(self.netD.state_dict(), '{}/pix2pix_D_epoch_{}'.format(output_dir, epoch))

    def save_image(self, epoch):
        output_image = torch.cat([self.realA, self.fakeB, self.realB], dim=3)
        vutils.save_image(output_image,
                '{}/pix2pix_epoch_{}.png'.format(self.config.output_dir, epoch),
                normalize=True)

    def print_epochloss(self, epoch, iteration):
        elapsed_time = time.time() - self.training_start_time
        
        losses = [self.epoch_lossG, self.epoch_lossG_GAN, self.epoch_lossG_L1, 
                  self.epoch_lossD, self.epoch_lossD_real, self.epoch_lossD_fake]
        for i, loss in enumerate(losses):
            losses[i] /= iteration
        message = 'epoch: {}, time: {:.3f}, lossG: {:.3f}, lossG_GAN: {:.3f}, lossG_L1: {:.3f}, lossD: {:.3f}, lossD_real: {:.3f}, lossD_fake: {:.3f}, lr: {:.5f}'.format(
            epoch, elapsed_time, *losses, self.optimizerG.param_groups[0]['lr'])
        print(message)

        self.append_log(message)
        self.epoch_lossG, self.epoch_lossG_GAN, self.epoch_lossG_L1 = 0.0, 0.0, 0.0,
        self.epoch_lossD, self.epoch_lossD_fake, self.epoch_lossD_real = 0.0, 0.0, 0.0

    def append_log(self, message):
        log_file = '{}/pix2pix.log'.format(self.config.output_dir)
        with open(log_file, "a") as log_file:
            log_file.write('{}\n'.format(message))  # save the message

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--epochs', type=int, default=200, help='epoch count')
    parser.add_argument('--save_data_interval', type=int, default=10, help='save data interval epochs')
    parser.add_argument('--save_image_interval', type=int, default=1, help='save image interval epochs')
    parser.add_argument('--log_interval', type=int, default=1, help='log interval epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='epoch count')
    parser.add_argument('--load_size', type=int, default=300, help='scale images to this size')
    parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
    parser.add_argument('--cpu', action='store_true', help='use cpu')
    parser.add_argument('--imgdir', default='./dataset/train/img', help='path to the data directory')
    parser.add_argument('--labeldir', default='./dataset/train/label', help='path to the data directory')
    parser.add_argument('--output_dir', default='./output', help='output directory')
    parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
    parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
    parser.add_argument('--generator', help='file path to data for generator')
    parser.add_argument('--discriminator', help='file path to data for discriminator')
    parser.add_argument('--epochs_lr_decay', type=int, default=100, help='epochs to delay lr to zero')
    parser.add_argument('--epochs_lr_decay_start', type=int, default=0, help='epochs to lr delay start')
    parser.add_argument('--direction', type=str, default='B2A', help='direction of conversion')
    args = parser.parse_args(args=['--imgdir', './dataset/train/img'])
    print(args)

    is_cpu = args.cpu or not torch.cuda.is_available()
    args.device_name = "cpu" if is_cpu else "cuda:0"
    args.device = torch.device(args.device_name)

    model = Pix2Pix(args)
    dataset = AlignedDataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    for epoch in range(1, args.epochs + 1):
        iteration = 0

        for i, data in enumerate(dataloader):
            model.train(data)
            iteration += len(data)

        if epoch % args.save_data_interval == 0:
            model.save(epoch)

        if epoch % args.save_image_interval == 0:
            model.save_image(epoch)

        if epoch % args.log_interval == 0:
            model.print_epochloss(epoch, iteration)

        model.update_learning_rate()