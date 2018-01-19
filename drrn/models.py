# -*- coding: utf-8 -*-
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Chain


class ResBlock(Chain):
    def __init__(self):
        super(ResBlock, self).__init__()
        with self.init_scope():
            self.cn1 = L.Convolution2D(None, 128, ksize=3, stride=1, pad=1)
            self.cn2 = L.Convolution2D(None, 3, ksize=3, stride=1, pad=1)
            self.bn1 = L.BatchNormalization(128)
            self.bn2 = L.BatchNormalization(3)

    def __call__(self, x: chainer.Variable):
        h = F.relu(self.bn1(self.cn1(x)))
        h = self.bn2(self.cn2(h))

        return h + x


class RecursiveBlock(Chain):
    def __init__(self, n_residual=3):
        super(RecursiveBlock, self).__init__()
        with self.init_scope():
            self.n_residual = n_residual
            self.conv = L.Convolution2D(None, 3, ksize=3, stride=1, pad=1)
            self.res = ResBlock()

    def __call__(self, x: chainer.Variable):
        h = F.relu(self.conv(x))
        for i in range(self.n_residual):
            h = self.res(h)

        return h + self.conv(x)


class DRRN(Chain):
    def __init__(self, n_recursive=3, n_residual=6):
        super(DRRN, self).__init__()
        with self.init_scope():
            self.n_recursive = n_recursive
            self.conv = L.Convolution2D(None, 3, ksize=3, stride=1, pad=1)
            self.recur = RecursiveBlock(n_residual=n_residual)

    def __call__(self, x: chainer.Variable):
        h = x
        for i in range(self.n_recursive):
            h = self.recur(h)

        h = self.conv(h)

        return h + x
