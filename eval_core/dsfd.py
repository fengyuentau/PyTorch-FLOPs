import cv2
import numpy as np
from tqdm import tqdm

from .base import Eval

class DSFD(Eval):
    '''Tests DSFD
    '''

    def __init__(self, dataset, model, ms=False, max_downsample=1):
        super(DSFD, self).__init__(dataset, model, ms, max_downsample)

    def _calc_shrink(self, h, w):
        max_im_shrink = (0x7fffffff / 200.0 / (h * w)) ** 0.5 # the max size of input image for caffe
        max_im_shrink = 3 if max_im_shrink > 3 else max_im_shrink
        shrink = max_im_shrink if max_im_shrink < 1 else 1
        return shrink, max_im_shrink

    def multi_scale_testing(self):


        flops_total = int(0)
        pbar = tqdm(self.dataset, desc='Multi-scale testing {} on {}'.format(self.model.name, self.dataset.name))
        for img in pbar:
            flops_det01, flops_det2, flops_det3, flops_det4 = [int(0)] * 4

            shrink, max_im_shrink = self._calc_shrink(img.shape[0], img.shape[1])

            # det0: scale 1, and det1: flip
            flops_det01 = self._calc_flops(img, shrink, flip=True)

            # det2
            st = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink
            flops_det2 = self._calc_flops(img, st, flip=False)
            if max_im_shrink > 0.75:
                flops_det2 += self._calc_flops(img, 0.75, flip=False)

            # det3
            bt = min(2, max_im_shrink) if max_im_shrink > 1 else (st + max_im_shrink) / 2
            flops_det3 = self._calc_flops(img, bt, flip=False)
            if max_im_shrink > 1.5:
                flops_det3 += self._calc_flops(img, 1.5, flip=False)
            if max_im_shrink > 2:
                bt *= 2
                while bt < max_im_shrink:
                    flops_det3 += self._calc_flops(img, bt, flip=False)
                    bt *= 2
                flops_det3 += self._calc_flops(img, max_im_shrink, flip=False)

            # det4
            flops_det4 = self._calc_flops(img, 0.25, flip=False)
            st = [1.25, 1.75, 2.25]
            for i in range(len(st)):
                if (st[i] <= max_im_shrink):
                    # cnt_s[i] += 1
                    flops_det4 += self._calc_flops(img, st[i], flip=False)

            flops_total += flops_det01 + flops_det2 + flops_det3 + flops_det4
        flops_avg = flops_total / len(self.dataset)
        return flops_avg