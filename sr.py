# -*- coding: utf-8 -*-
import argparse
import cv2
import numpy as np
from PIL import Image

import chainer

from drrn import models


def clip_img(x):
    return np.uint8(0 if x < 0 else (255 if x > 255 else x))


def resize_copy(img):
    dst = np.zeros((img.shape[0] * 4, img.shape[1] * 4, img.shape[2]), dtype=img.dtype)
    for x in range(4):
        for y in range(4):
            dst[x::4, y::4, :] = img
    return dst


def img2variable(img):
    return chainer.Variable(np.array([img.transpose(2, 0, 1)], dtype=np.float32))


def variable2img(x):
    print(x.data.max())
    print(x.data.min())
    img = (np.vectorize(clip_img)(x.data[0, :, :, :])).transpose(1, 2, 0)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    model = models.DRRN()
    chainer.serializers.load_npz(args.model, model)

    img = np.asarray(Image.open(args.image), dtype=np.float32)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    img_variable = img2variable(img)
    img_variable_sr = model(img_variable)
    img_sr = variable2img(img_variable_sr)

    cv2.imwrite("./result.png", cv2.cvtColor(img_sr, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
