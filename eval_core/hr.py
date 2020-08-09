import cv2
import numpy as np
from tqdm import tqdm

from .base import Eval

class HR(Eval):
    '''Tests HR
    '''

    def __init__(self, dataset, model, ms=False, max_downsample=1):
        super(HR,self).__init__(dataset, model, ms, max_downsample)

    def multi_scale_testing(self):
        scales = [-2, -1, 0, 1]

        flops_total = int(0)
        pbar = tqdm(self.dataset, desc='Multi-scale testing {} on {}'.format(self.model.name, self.dataset.name))
        for img in pbar:
            flops = int(0)
            for s in scales:
                flops += self._calc_flops(img, 2.0**s, flip=False)
            flops_total += flops
        flops_avg = flops_total / len(self.dataset)
        return flops_avg