import cv2
import numpy as np

import flops_counter

from tqdm import tqdm

def calc_shrink(height, width):
    """
    Args:
        height (int): image height.
        width (int): image width.
    """
    # avoid out of memory
    max_shrink_v1 = (0x7fffffff / 577.0 / (height * width))**0.5
    max_shrink_v2 = ((678 * 1024 * 2.0 * 2.0) / (height * width))**0.5

    def get_round(x, loc):
        str_x = str(x)
        if '.' in str_x:
            str_before, str_after = str_x.split('.')
            len_after = len(str_after)
            if len_after >= 3:
                str_final = str_before + '.' + str_after[0:loc]
                return float(str_final)
            else:
                return x

    max_shrink = get_round(min(max_shrink_v1, max_shrink_v2), 2) - 0.3
    if max_shrink >= 1.5 and max_shrink < 2:
        max_shrink = max_shrink - 0.1
    elif max_shrink >= 2 and max_shrink < 3:
        max_shrink = max_shrink - 0.2
    elif max_shrink >= 3 and max_shrink < 4:
        max_shrink = max_shrink - 0.3
    elif max_shrink >= 4 and max_shrink < 5:
        max_shrink = max_shrink - 0.4
    elif max_shrink >= 5:
        max_shrink = max_shrink - 0.5

    shrink = max_shrink if max_shrink < 1 else 1
    return shrink, max_shrink

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

        # det3
        bt = min(2, max_im_shrink) if max_im_shrink > 1 else (st + max_im_shrink) / 2
        flops_det3 = calc_flops(model, img, bt, flip=False)
        if max_im_shrink > 2:
            bt *= 2
            while bt < max_im_shrink:
                flops_det3 += calc_flops(model, img, bt, flip=False)
                bt *= 2
            flops_det3 += calc_flops(model, img, max_im_shrink, flip=False)

        # det4
        flops_det4 = calc_flops(model, img, 0.25, flip=False)
        st = [0.75, 1.25, 1.5, 1.75] # [0.75, 1.25, 1.5, 1.75]
        for i in range(len(st)):
            if (st[i] <= max_im_shrink):
                flops_det4 += calc_flops(model, img, st[i], flip=False)

        flops_total += flops_det01 + flops_det2 + flops_det3 + flops_det4
    flops_avg = flops_total / len(dataset)
    return flops_avg