# -*- coding: utf-8 -*-
import functools
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Chain


class ResBlock(Chain):
    def __init__(self):
        super(ResBlock, self).__init__()
        with self.init_scope():
            self.cn1 = L.Convolution2D(None, 64, ksize=3, stride=1, pad=1)
            self.cn2 = L.Convolution2D(None, 64, ksize=3, stride=1, pad=1)
            self.bn1 = L.BatchNormalization(64)
            self.bn2 = L.BatchNormalization(64)

    def __call__(self, x: chainer.Variable):
        h = F.relu(self.bn1(self.cn1(x)))
        h = self.bn2(self.cn2(h))

        return h + x


class RecursiveBlock(Chain):
    def __init__(self, n_residual=3):
        super(RecursiveBlock, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(None, 64, ksize=3, stride=1, pad=1)

            self.resblocks = []
            for i in range(n_residual):
                self.resblocks.append(ResBlock())

    def __call__(self, x: chainer.Variable):
        h = F.relu(self.conv(x))
        for res in self.resblocks:
            h = res(h)

        return h + self.conv(x)


class DRRN(Chain):
    def __init__(self, n_recursive=5):
        super(DRRN, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(None, 3, ksize=3, stride=1, pad=1)
            self.deconv = L.Deconvolution2D(3, 75, stride=1, pad=1)

            self.recurblocks = []
            for i in range(n_recursive):
                self.recurblocks.append(RecursiveBlock())

    def __call__(self, x: chainer.Variable):
        h = x
        for recur in self.recurblocks:
            h = recur(h)

        h = self.conv(h)

        return h + x
