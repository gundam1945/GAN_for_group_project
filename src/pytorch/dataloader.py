import argparse
import torch
from torchvision import datasets
import torchvision.transforms as transforms


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image samples")
parser.add_argument("--normal_mean", type=float, default=0, help="mean to be subtracted from normal")
parser.add_argument("--normal_sd", type=float, default=1, help="sd to be divided by difference of normal and mean")
opt = parser.parse_args()
print(opt)


def get_MNIST_set(opt, train):
    return torch.utils.data.DataLoader(
        datasets.MNIST(
            "../../data",
            train=train,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(28), transforms.ToTensor(), transforms.Normalize([opt.normal_mean], [opt.normal_sd])]
            ),
        ),
        batch_size=opt.batch_size,
        shuffle=True,
    )


def get_CIFAR_set(opt, train):
    return torch.utils.data.DataLoader(
        datasets.CIFAR10(
            "../../data",
            train=train,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(32), transforms.ToTensor(), transforms.Normalize([opt.normal_mean, opt.normal_mean, opt.normal_mean], [opt.normal_sd, opt.normal_sd, opt.normal_sd])]
            ),
        ),
        batch_size=opt.batch_size,
        shuffle=True,
    )


# def get_TF_set(opt, train):
#     return torch.utils.data.DataLoader(
#         datasets.(
#             "../../data",
#             train=train,
#             download=True,
#             transform=transforms.Compose(
#                 [transforms.Resize(32), transforms.ToTensor(), transforms.Normalize([opt.mean, opt.mean, opt.mean], [opt.sd, opt.sd, opt.sd])]
#             ),
#         ),
#         batch_size=opt.batch_size,
#         shuffle=True,
#     )


if __name__ == "__main__":
    get_MNIST_set(opt, True)
    print("MAIN END")