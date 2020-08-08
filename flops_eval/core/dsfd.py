import cv2
import numpy as np

import flops_counter

from tqdm import tqdm

def calc_shrink(h, w):
    max_im_shrink = (0x7fffffff / 200.0 / (h * w)) ** 0.5 # the max size of input image for caffe
    max_im_shrink = 3 if max_im_shrink > 3 else max_im_shrink
    shrink = max_im_shrink if max_im_shrink < 1 else 1
    return shrink, max_im_shrink

def calc_flops(model, img, shrink=1, flip=False, max_downsample=16):
    x = img
    if shrink != 1:
        x = cv2.resize(img, None, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_LINEAR)
    h, w, c = x.shape

    x = flops_counter.TensorSize([1, c, h, w])

    model(x)
    flops = model.flops * 2 if flip else model.flops

    model.set_flops_zero()
    return flops

def flops_eval(dataset, model):


    flops_total = int(0)
    pbar = tqdm(dataset, desc='Evaluating {} on {}'.format(model.name, dataset.name))
    for img in pbar:
        flops_det01, flops_det2, flops_det3, flops_det4 = [int(0)] * 4

        shrink, max_im_shrink = calc_shrink(img.shape[0], img.shape[1])

        # det0: scale 1, and det1: flip
        flops_det01 = calc_flops(model, img, shrink, flip=True)

        # det2
        st = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink
        flops_det2 = calc_flops(model, img, st, flip=False)
        if max_im_shrink > 0.75:
            flops_det2 += calc_flops(model, img, 0.75, flip=False)

        # det3
        bt = min(2, max_im_shrink) if max_im_shrink > 1 else (st + max_im_shrink) / 2
        flops_det3 = calc_flops(model, img, bt, flip=False)
        if max_im_shrink > 1.5:
            flops_det3 += calc_flops(model, img, 1.5, flip=False)
        if max_im_shrink > 2:
            bt *= 2
            while bt < max_im_shrink:
                flops_det3 += calc_flops(model, img, bt, flip=False)
                bt *= 2
            flops_det3 += calc_flops(model, img, max_im_shrink, flip=False)

        # det4
        flops_det4 = calc_flops(model, img, 0.25, flip=False)
        st = [1.25, 1.75, 2.25]
        for i in range(len(st)):
            if (st[i] <= max_im_shrink):
                # cnt_s[i] += 1
                flops_det4 += calc_flops(model, img, st[i], flip=False)

        flops_total += flops_det01 + flops_det2 + flops_det3 + flops_det4
    flops_avg = flops_total / len(dataset)
    return flops_avg