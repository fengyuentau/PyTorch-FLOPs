import cv2
import numpy as np

import flops_counter

from tqdm import tqdm

def calc_flops(model, img, scale=1.0, flip=True, max_downsample=16):
    img_s = img
    if scale != 1.0:
        img_s = cv2.resize(img, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    img_s_h, img_s_w, img_s_c = img_s.shape
    # skip when input resolution is too huge, see line 268 in cnn_widerface_test_AB.m
    if scale > 1 and (img_s_h > 5000 or img_s_w > 5000):
        return 0

    x = flops_counter.TensorSize([1, img_s_c, img_s_h, img_s_w])

    model(x)
    flops = model.flops * 2 if flip else model.flops

    model.set_flops_zero()
    return flops

def flops_eval(dataset, model):
    scales = [-2, -1, 0, 1]

    flops_total = int(0)
    pbar = tqdm(dataset, desc='Evaluating {} on {}'.format(model.name, dataset.name))
    for img in pbar:
        flops = int(0)
        for s in scales:
            flops += calc_flops(model, img, 2.0**s, flip=False)
        flops_total += flops
    flops_avg = flops_total / len(dataset)
    return flops_avg