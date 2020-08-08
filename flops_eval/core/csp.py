import cv2
import numpy as np

import flops_counter

from tqdm import tqdm

def _calc_shrink(h, w):
    max_im_shrink = (0x7fffffff / 577.0 / (h * w)) ** 0.5  # the max size of input image
    shrink = max_im_shrink if max_im_shrink < 1 else 1
    return shrink, max_im_shrink

def _calc_flops(model, img, scale=1, flip=False, max_downsample=16):
    img_h, img_w = img.shape[:2]
    img_h_new, img_w_new = int(np.ceil(scale * img_h / 16) * 16), int(np.ceil(scale * img_w / 16) * 16)
    scale_h, scale_w = img_h_new / img_h, img_w_new / img_w

    img_s = cv2.resize(img, None, None, fx=scale_w, fy=scale_h, interpolation=cv2.INTER_LINEAR)

    img_s_h, img_s_w, img_s_c = img_s.shape
    x = flops_counter.TensorSize([1, img_s_c, img_s_h, img_s_w])

    model(x)
    flops = model.flops * 2 if flip else model.flops

    model.set_flops_zero()
    return flops

def flops_eval(dataset, model):
    st = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0, 2.25]

    flops_total = 0
    pbar = tqdm(dataset, desc='Evaluating {} on {}'.format(model.name, dataset.name))
    for img in pbar:
        flops = int(0)

        shrink, max_im_shrink = _calc_shrink(img.shape[0], img.shape[1])
        for scale in st:
            if scale in [0.25, 0.5, 0.75, 1]:
                flops += _calc_flops(model, img, scale, flip=True)
                # pass
            elif scale <= max_im_shrink:
                flops += _calc_flops(model, img, scale, flip=True)
            
        flops_total += flops
    flops_avg = flops_total / len(dataset)
    # print('Average FLOPs: {:.2e} GFLOPs'.format(flops_avg))
    return flops_avg