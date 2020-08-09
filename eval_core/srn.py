import math

import cv2
import numpy as np
from tqdm import tqdm

from .base import Eval

class SRN(Eval):
    '''Tests SRN
    '''

    def __init__(self, dataset, model, ms=False, max_downsample=1):
        super(SRN, self).__init__(dataset, model, ms, max_downsample)

    def multi_scale_testing(self):
        largest_input = 2100 * 2100

        flops_total = int(0)
        pbar = tqdm(self.dataset, desc='Multi-scale testing {} on {}'.format(self.model.name, self.dataset.name))
        for img in pbar:
            flops_det01, flops_det2, flops_det3, flops_det4 = [int(0)] * 4

            img_h, img_w, img_c = img.shape
            if img_h * img_w < largest_input:
                # det0 and det1 (0-original, 1-flip)
                flops_det01 = self._calc_flops(img, 1, flip=False)
                # det2 (shrink 0.5)
                flops_det2 = self._calc_flops(img, 0.5, flip=False)
                # det3 (enlarge)
                enlarge_time = int(math.floor(math.log(largest_input / img_w / img_h, 2.25)))
                for t in range(enlarge_time):
                    resize_scale = math.pow(1.5, t+1)
                    flops_det3 += self._calc_flops(img, resize_scale, flip=False)
                # det4 (final ratio)
                final_ratio = math.sqrt(largest_input / img_h / img_w)
                flops_det4 = self._calc_flops(img, final_ratio, flip=False)
            else:
                largest_ratio = math.sqrt(largest_input / img_w / img_h)
                # det0 and det1 (0-largest, 1-largest's flip)
                flops_det01 = self._calc_flops(img, largest_ratio, flip=False)
                # det2 (shrink 0.75)
                flops_det2 = self._calc_flops(img, 0.75, flip=False)
                # det3 (shrink 0.5)
                flops_det3 = self._calc_flops(img, 0.5, flip=False)
                # det4 (no det4)

            flops_total += flops_det01 + flops_det2 + flops_det3 + flops_det4
        flops_avg = flops_total / len(self.dataset)
        return flops_avg