import math

import cv2
import numpy as np

import flops_counter

from tqdm import tqdm

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
    largest_input = 2100 * 2100

    flops_total = int(0)
    pbar = tqdm(dataset, desc='Evaluating {} on {}'.format(model.name, dataset.name))
    for img in pbar:
        flops_det01, flops_det2, flops_det3, flops_det4 = [int(0)] * 4

        img_h, img_w, img_c = img.shape
        if img_h * img_w < largest_input:
            # det0 and det1 (0-original, 1-flip)
            flops_det01 = calc_flops(model, img, 1, flip=False)
            # det2 (shrink 0.5)
            flops_det2 = calc_flops(model, img, 0.5, flip=False)
            # det3 (enlarge)
            enlarge_time = int(math.floor(math.log(largest_input / img_w / img_h, 2.25)))
            for t in range(enlarge_time):
                resize_scale = math.pow(1.5, t+1)
                flops_det3 += calc_flops(model, img, resize_scale, flip=False)
            # det4 (final ratio)
            final_ratio = math.sqrt(largest_input / img_h / img_w)
            flops_det4 = calc_flops(model, img, final_ratio, flip=False)
        else:
            largest_ratio = math.sqrt(largest_input / img_w / img_h)
            # det0 and det1 (0-largest, 1-largest's flip)
            flops_det01 = calc_flops(model, img, largest_ratio, flip=False)
            # det2 (shrink 0.75)
            flops_det2 = calc_flops(model, img, 0.75, flip=False)
            # det3 (shrink 0.5)
            flops_det3 = calc_flops(model, img, 0.5, flip=False)
            # det4 (no det4)

        flops_total += flops_det01 + flops_det2 + flops_det3 + flops_det4
    flops_avg = flops_total / len(dataset)
    return flops_avg