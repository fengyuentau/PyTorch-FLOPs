import cv2
import numpy as np

import flops_counter

from tqdm import tqdm

def calc_shrink(h, w):
    max_im_shrink = (0x7fffffff / 577.0 / (h * w)) ** 0.5 # the max size of input image for caffe
    shrink = max_im_shrink if max_im_shrink < 1 else 1
    return shrink, max_im_shrink

def calc_flops(model, img, scale=1, flip=False, max_downsample=16):
    img_h, img_w, img_c = img.shape[:3]
    if scale != 1:
        img_h_new, img_w_new = int(np.ceil(scale * img_h / 16) * 16), int(np.ceil(scale * img_w / 16) * 16)
        scale_h, scale_w = img_h_new / img_h, img_w_new / img_w
        img_s = cv2.resize(img, None, None, fx=scale_w, fy=scale_h, interpolation=cv2.INTER_LINEAR)
        img_h, img_w, img_c = img_s.shape

    x = flops_counter.TensorSize([1, img_c, img_h, img_w])

    model(x)
    flops = model.flops * 2 if flip else model.flops

    model.set_flops_zero()
    return flops

def flops_eval(dataset, model):


    flops_total = int(0)
    pbar = tqdm(dataset, desc='Evaluating {} on {}'.format(model.name, dataset.name))
    for img in pbar:
        flops_det01, flops_det2, flops_det3 = [int(0)] * 3

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
            while bt < max_im_shrink: # triger twice at val set, trigger 10 times at test set
                flops_det3 += calc_flops(model, img, bt, flip=False)
                bt *= 2
            flops_det3 += calc_flops(model, img, max_im_shrink, flip=False)

        flops_total += flops_det01 + flops_det2 + flops_det3
    flops_avg = flops_total / len(dataset)
    return flops_avg