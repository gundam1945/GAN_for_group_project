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

from model4 import Model, get_dataloader


os.makedirs("images", exist_ok=True)


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image samples")
parser.add_argument("--normal_mean", type=float, default=0, help="mean to be subtracted from normal")
parser.add_argument("--normal_sd", type=float, default=1, help="sd to be divided by difference of normal and mean")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def main():
    # Initialize generator and discriminator
    model = Model(opt)
    generator = model.generator
    discriminator = model.discriminator

    if os.path.isfile(model.PATH_G):
        generator.load_state_dict(torch.load(model.PATH_G))

    if os.path.isfile(model.PATH_D):
        discriminator.load_state_dict(torch.load(model.PATH_D))

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # Configure data loader
    os.makedirs("../../data/mnist", exist_ok=True)

    now = (datetime.datetime.now())

    train_model(generator, discriminator, optimizer_G, optimizer_D, adversarial_loss)

    print(now)
    print(datetime.datetime.now())

    torch.save(generator.state_dict(), model.PATH_G)
    torch.save(discriminator.state_dict(), model.PATH_D)


def generate_sample(batch_size):
    model = Model(opt)
    generator = model.generator

    if os.path.isfile(model.PATH_G):
        generator.load_state_dict(torch.load(model.PATH_G))

    if cuda:
        generator.cuda()

    # Sample noise as generator input
    z = Variable(Tensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))

    # Generate a batch of images
    gen_imgs = generator(z)

    return gen_imgs


def train_model(generator, discriminator, optimizer_G, optimizer_D, adversarial_loss):
    # ----------
    #  Training
    # ----------
    dataloader = get_dataloader(opt)

    for epoch in range(opt.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            # Adversarial ground truths
            valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

            # Generate a batch of images
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            if i % 100 == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch+1, opt.n_epochs, i + 1, len(dataloader), d_loss.item(), g_loss.item())
                )

            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)


if __name__ == "__main__":
    main()