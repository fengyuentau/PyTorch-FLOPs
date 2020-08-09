import cv2
import numpy as np
from tqdm import tqdm

from .base import Eval

class SHF(Eval):
    '''Tests SHF
    '''

    def __init__(self, dataset, model, ms=False, max_downsample=16):
        max_downsample = 16
        super(SHF, self).__init__(dataset, model, ms, max_downsample)

    def _calc_shrink(self, h, w):
        target_size, max_size = [800, 1200]
        im_size_min = np.min([h, w])
        im_size_max = np.max([h, w])
        im_scale = float(target_size) / float(im_size_min)
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)

        pyramid_scales = [
            float(scale) / 800 * im_scale
            for scale in [100, 300, 600, 1000, 1400]
        ]
        return pyramid_scales

    def multi_scale_testing(self):
        flops_total = int(0)
        pbar = tqdm(self.dataset, desc='Multi-scale testing {} on {}'.format(self.model.name, self.dataset.name))
        for img in pbar:
            flops = int(0)

            pyramid_scales = self._calc_shrink(img.shape[0], img.shape[1])

            for ps in pyramid_scales:
                flops += self._calc_flops(img, ps, flip=True)

            flops_total += flops
        flops_avg = flops_total / len(self.dataset)
        return flops_avg