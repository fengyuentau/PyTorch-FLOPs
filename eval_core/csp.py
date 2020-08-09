import cv2
import numpy as np
from tqdm import tqdm

from .base import Eval

class CSP(Eval):
    '''Test CSP
    '''

    def __init__(self, dataset, model, ms=False, max_downsample=16):
        max_downsample = 16
        super(CSP, self).__init__(dataset, model, ms, max_downsample)

    def _calc_shrink(self, h, w):
        max_im_shrink = (0x7fffffff / 577.0 / (h * w)) ** 0.5  # the max size of input image
        shrink = max_im_shrink if max_im_shrink < 1 else 1
        return shrink, max_im_shrink

    def multi_scale_testing(self):
        st = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0, 2.25]

        flops_total = 0
        pbar = tqdm(self.dataset, desc='Multi-scale testing {} on {}'.format(self.model.name, self.dataset.name))
        for img in pbar:
            flops = int(0)

            shrink, max_im_shrink = self._calc_shrink(img.shape[0], img.shape[1])
            for scale in st:
                if scale in [0.25, 0.5, 0.75, 1]:
                    flops += self._calc_flops(img, scale, flip=True)
                    # pass
                elif scale <= max_im_shrink:
                    flops += self._calc_flops(img, scale, flip=True)
                
            flops_total += flops
        flops_avg = flops_total / len(self.dataset)
        return flops_avg