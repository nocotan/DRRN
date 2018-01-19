# -*- coding: utf-8 -*-
import argparse
import glob
import logging
import os
import scipy.misc
from tqdm import tqdm

import numpy
import chainer
import chainer.functions as F
from chainer.optimizers import Adam
from chainer.iterators import MultiprocessIterator

from drrn import datasets
from drrn import models


def forward(x, y, model):
    t = model(x)
    loss = F.mean_squared_error(t, y)
    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--batchsize", type=int, default=10)
    parser.add_argument("--outdirname", required=True)
    parser.add_argument("--recursive", type=int, default=3)
    args = parser.parse_args()

    OUTPUT_DIRECTORY = args.outdirname
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)

    # GPU
    if args.gpu > 0:
        chainer.cuda.check_cuda_available()
        chainer.cuda.get_device(args.gpu).use()
        xp = chainer.cuda.cupy
    else:
        xp = numpy

    # load dataset
    paths = glob.glob(args.dataset)
    dataset = datasets.PreprocessedImageDataset(
        paths=paths,
        cropsize=96, resize=(300, 300)
    )

    iterator = MultiprocessIterator(dataset,
                                    batch_size=args.batchsize,
                                    repeat=True,
                                    shuffle=True)

    model = models.DRRN()
    if args.gpu >= 0:
        model.to_gpu()

    optimizer = Adam()
    optimizer.setup(model)

    for epoch in range(50):
        for zipped_batch in tqdm(iterator):
            lr = chainer.Variable(xp.array([zipped[0] for zipped in zipped_batch]))
            hr = chainer.Variable(xp.array([zipped[1] for zipped in zipped_batch]))

            loss = forward(lr, hr, model)
            optimizer.update(forward, lr, hr, model)


        print("Epoch: {}, Loss: {}".format(epoch, loss.data))
        sr = numpy.array(model(lr).data)[0]
        sr = sr.reshape(96, 96, 3)
        scipy.misc.imsave("output/out.png", sr)


if __name__ == '__main__':
    main()
