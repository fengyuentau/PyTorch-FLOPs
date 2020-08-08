import math

import cv2
import numpy as np

import flops_counter

from tqdm import tqdm

def calc_scale(image_shape, target_size, max_size):
    im_size_min = min(image_shape)
    im_size_max = max(image_shape)

    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if int(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    return im_scale

def calc_flops(model, image, shrink=1, flip=False, max_downsample=16):
    image_shape = image.shape
    if shrink != 1:
        x = cv2.resize(image, None, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_LINEAR)
        image_shape = x.shape

    h, w, c = image_shape
    x = flops_counter.TensorSize([1, c, h, w])

    model(x)
    flops = model.flops * 2 if flip else model.flops

    model.set_flops_zero()
    return flops

def flops_eval(dataset, model):
    scales = [500, 800, 1200, 1600, ]
    pyramid_base_size = [800,1200]

    flops_total = int(0)
    pbar = tqdm(dataset, desc='Evaluating {} on {}'.format(model.name, dataset.name))
    for img in pbar:
        flops = int(0)

        base_scale = calc_scale(img.shape[0:2], pyramid_base_size[0], pyramid_base_size[1])
        pyramid_scales = [ float(scale)/pyramid_base_size[0]*base_scale for scale in scales]
        for pyramid_scale in pyramid_scales:
            flops += calc_flops(model, img, pyramid_scale)

        flops_total += flops
    flops_avg = flops_total / len(dataset)
    return flops_avg