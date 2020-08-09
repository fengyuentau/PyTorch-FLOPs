import cv2
import numpy as np
from tqdm import tqdm

from .base import Eval

class S3FD(Eval):
    '''Tests S3FD
    '''

    def __init__(self, dataset, model, ms=False, max_downsample=16):
        max_downsample = 16
        super(S3FD, self).__init__(dataset, model, ms, max_downsample)

    def _calc_shrink(self, h, w):
        max_im_shrink = (0x7fffffff / 577.0 / (h * w)) ** 0.5 # the max size of input image for caffe
        shrink = max_im_shrink if max_im_shrink < 1 else 1
        return shrink, max_im_shrink

    def multi_scale_testing(self):


        flops_total = int(0)
        pbar = tqdm(self.dataset, desc='Multi-scale testing {} on {}'.format(self.model.name, self.dataset.name))
        for img in pbar:
            flops_det01, flops_det2, flops_det3 = [int(0)] * 3

            shrink, max_im_shrink = self._calc_shrink(img.shape[0], img.shape[1])

            # det0: scale 1, and det1: flip
            flops_det01 = self._calc_flops(img, shrink, flip=True)

            # det2
            st = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink
            flops_det2 = self._calc_flops(img, st, flip=False)

            # det3
            bt = min(2, max_im_shrink) if max_im_shrink > 1 else (st + max_im_shrink) / 2
            flops_det3 = self._calc_flops(img, bt, flip=False)
            if max_im_shrink > 2:
                bt *= 2
                while bt < max_im_shrink: # triger twice at val set, trigger 10 times at test set
                    flops_det3 += self._calc_flops(img, bt, flip=False)
                    bt *= 2
                flops_det3 += self._calc_flops(img, max_im_shrink, flip=False)

            flops_total += flops_det01 + flops_det2 + flops_det3
        flops_avg = flops_total / len(self.dataset)
        return flops_avg