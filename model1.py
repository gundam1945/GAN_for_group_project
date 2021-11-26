import argparse
import datetime
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch


class Model(object):
    def __init__(self, opt):
        self.img_shape = (opt.channels, opt.img_size, opt.img_size)
        dir = "\\state\\gan1"
        os.makedirs(os.path.abspath(os.path.curdir) + dir, exist_ok=True)
        self.PATH_G = os.path.abspath(os.path.curdir) + dir + "\\generator.pth"
        self.PATH_D = os.path.abspath(os.path.curdir) + dir + "\\discriminator.pth"
        self.generator = Model.Generator(opt)
        self.discriminator = Model.Discriminator(opt)

    class Generator(nn.Module):
        def __init__(self, opt):
            super(Model.Generator, self).__init__()
            self.img_shape = (opt.channels, opt.img_size, opt.img_size)

            def block(in_feat, out_feat, normalize=True):
                layers = [nn.Linear(in_feat, out_feat)]
                if normalize:
                    layers.append(nn.BatchNorm1d(out_feat, 0.8))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                return layers

            self.model = nn.Sequential(
                *block(opt.latent_dim, 128, normalize=False),
                *block(128, 256),
                *block(256, 512),
                *block(512, 1024),
                nn.Linear(1024, int(np.prod(self.img_shape))),
                nn.Tanh()
            )

        def forward(self, z):
            img = self.model(z)
            img = img.view(img.size(0), *self.img_shape)
            return img

    class Discriminator(nn.Module):
        def __init__(self, opt):
            super(Model.Discriminator, self).__init__()
            self.img_shape = (opt.channels, opt.img_size, opt.img_size)

            self.model = nn.Sequential(
                nn.Linear(int(np.prod(self.img_shape)), 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 1),
                nn.Sigmoid(),
            )

        def forward(self, img):
            img_flat = img.view(img.size(0), -1)
            validity = self.model(img_flat)

            return validity


def get_dataloader(opt):
    return torch.utils.data.DataLoader(
        datasets.MNIST(
            "../../data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=opt.batch_size,
        shuffle=True,
    )