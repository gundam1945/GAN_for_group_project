import os
import numpy
import pandas as pd
import sys
import time

import aesara
from collections import OrderedDict
T = aesara.tensor

import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


BATCH_SIZE = 100
SAMPLE_SIZE = 1000


#region ported from ll.py

def get_noise(size):
    from theano.sandbox import rng_mrg
    return rng_mrg.MRG_RandomStreams(2014 * 5 + 27).uniform(low=-numpy.sqrt(3), high=numpy.sqrt(3), size=size, dtype='float32')


def sharedX(value, name=None, borrow=False, dtype=None):
    """
    Transform value into a shared variable of type floatX
    Parameters
    ----------
    value : WRITEME
    name : WRITEME
    borrow : WRITEME
    dtype : str, optional
        data type. Default value is theano.config.floatX
    Returns
    -------
    WRITEME
    """

    import theano
    if dtype is None:
        dtype = theano.config.floatX
    return theano.shared(theano._asarray(value, dtype=dtype),
                         name=name,
                         borrow=borrow)


def get_avg_ll(trainset, testset, sigma):
    import theano
    # _, model_path, sigma = sys.argv
    # from pylearn2.utils import serial
    # model = serial.load(model_path)
    # from pylearn2.config import yaml_parse
    # dataset = yaml_parse.load(model.dataset_yaml_src)
    # dataset = dataset.get_test_set()
    # from pylearn2.utils import sharedX
    # g = model.generator
    # n = g.get_input_space().get_total_dimension()
    # X = sharedX(dataset.X)
    # m = dataset.X.shape[0]
    # accumulator = sharedX(np.zeros((m,)))
    X = sharedX(testset.data[0:10000].reshape(10000, 784))
    m = testset.data[0:10000].reshape(10000, 784).shape[0]
    accumulator = sharedX(numpy.zeros((m,)))
    #z_samples = g.get_noise(1)
    #x_samples = g.mlp.fprop(z_samples)
    x_samples = sharedX(trainset.data[0:1].reshape(784))
    #x_samples = get_noise(tuple([1]))
    updates = OrderedDict()
    num_samples = theano.shared(1)
    sigma = sharedX(float(sigma))
    prev = accumulator
    cur = -0.5 * T.sqr(X - x_samples).sum(axis=1) / T.sqr(sigma)
    ofs = T.maximum(prev, cur)
    num_samples_f = T.cast(num_samples, 'float32')
    updates[accumulator] = ofs + T.log(num_samples_f * T.exp(prev - ofs) + T.exp(cur - ofs)) - T.log(num_samples_f + 1.)
    updates[num_samples] = num_samples + 1
    f = theano.function([], updates=updates)
    updates[accumulator] = cur
    del updates[num_samples]
    first = theano.function([], updates=updates)
    avg_ll = accumulator.mean() - 0.5 * X.shape[1] * T.log(2 * numpy.pi * T.sqr(sigma))

    import time
    prev_t = time.time()
    first()
    while True:
        v = avg_ll.eval()
        i = num_samples.get_value()
        if i == 1 or i % 1000 == 0:
            now_t = time.time()
            print(i, v, now_t - prev_t)
            prev_t = now_t
        if numpy.isnan(v) or numpy.isinf(v):
            break
        f()

#endregion

#region ported from parzen_ll.py

def log_mean_exp(a):
    """
    Credit: Yann N. Dauphin
    """

    max_ = a.max(1)

    return max_ + T.log(T.exp(a - max_.dimshuffle(0, 'x')).mean(1))


def get_nll(x, parzen, batch_size=10):
    """
    Credit: Yann N. Dauphin
    """

    inds = range(x.shape[0])
    n_batches = int(numpy.ceil(float(len(inds)) / batch_size))

    times = []
    nlls = []
    for i in range(n_batches):
        begin = time.time()
        nll = parzen(x[inds[i::n_batches]])
        end = time.time()
        times.append(end-begin)
        nlls.extend(nll)

    return numpy.array(nlls)


def theano_parzen(mu, sigma):
    """
    Credit: Yann N. Dauphin
    """

    x = aesara.tensor.matrix()
    mu = aesara.shared(mu)
    a = ( x.dimshuffle(0, 'x', 1) - mu.dimshuffle('x', 0, 1) ) / sigma
    E = log_mean_exp(-0.5*(a**2).sum(2))
    Z = mu.shape[1] * T.log(sigma * numpy.sqrt(numpy.pi * 2))

    return aesara.function([x], E - Z)


def cross_validate_sigma(samples, data, sigmas, batch_size):

    lls = []
    for sigma in sigmas:
        parzen = theano_parzen(samples, sigma)
        tmp = get_nll(data, parzen, batch_size = batch_size)
        num = numpy.asarray(tmp).mean()
        print(sigma, num)
        lls.append(num)
        del parzen

    ind = numpy.argmax(lls)
    return sigmas[ind]


def get_mean_nll(trainset, testset, sigma = 0):
    samples = numpy.asarray((trainset.data[0:SAMPLE_SIZE]/256-0.5).reshape(SAMPLE_SIZE, 784), dtype=aesara.config.floatX)
    valid = numpy.asarray((trainset.data[50000:51000]/256-0.5).reshape(1000, 784), dtype=aesara.config.floatX)
    test = numpy.asarray((testset.data/256-0.5).reshape(10000, 784), dtype=aesara.config.floatX)

    if sigma == 0:
        sigma_range = numpy.logspace(-1, 1, num=20)
        sigma = cross_validate_sigma\
                (
                    samples,
                    valid,
                    sigma_range,
                    BATCH_SIZE
                )

    parzen = theano_parzen(samples, sigma)
    ll = get_nll(test, parzen, batch_size = BATCH_SIZE)
    se = ll.std() / numpy.sqrt(test.shape[0])

    print("Log-Likelihood of test set = {}, se: {}".format(ll.mean(), se))


def get_sigma(trainset, testset):
    samples = numpy.asarray((trainset.data[0:SAMPLE_SIZE]/256-0.5).reshape(SAMPLE_SIZE, 784), dtype=aesara.config.floatX)
    valid = numpy.asarray((trainset.data[50000:51000]/256-0.5).reshape(1000, 784), dtype=aesara.config.floatX)
    test = numpy.asarray((testset.data/256-0.5).reshape(10000, 784), dtype=aesara.config.floatX)

    sigma_range = numpy.logspace(-1, 0, num=200)
    sigma = cross_validate_sigma\
            (
                samples,
                valid,
                sigma_range,
                BATCH_SIZE
            )
    return sigma


#endregion


def get_dataset(name):
    if name == "MNIST":
        return datasets\
                   .MNIST("../../data/mnist", train=True,
                       download=True,
                       transform=transforms.Compose(
                           [transforms.Resize(28),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5], [0.5])]
                       ),
            ), datasets\
                   .MNIST("../../data/mnist", train=False,
                       download=True,
                       transform=transforms.Compose(
                           [transforms.Resize(28),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5], [0.5])]
                       ),
            )
    else:
        sys.exit(0)


def get_dataloader(set):
    return torch.utils.data.DataLoader(set,batch_size=BATCH_SIZE,shuffle=True,)


def main():
    trainset, testset = get_dataset("MNIST")
    #sigma = get_sigma(trainset, testset)
    sigma = 0.2
    get_mean_nll(trainset, testset, sigma)
    # get_avg_ll(trainset, testset, sigma)

    print("MAIN END")


if __name__ == "__main__":
    main()