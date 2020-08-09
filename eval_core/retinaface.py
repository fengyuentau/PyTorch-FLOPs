import cv2
import numpy as np
from tqdm import tqdm

from .base import Eval

class RetinaFace(Eval):
    '''Tests RetinaFace
    '''

    def __init__(self, dataset, model, ms=False, max_downsample=16):
        super(RetinaFace, self).__init__(dataset, model, ms, max_downsample)

    def _calc_shrink(self, h, w):
        TEST_SCALES = [500, 800, 1100, 1400, 1700]
        target_size = 800
        max_size = 1200
        im_size_min = np.min([h, w])
        im_size_max = np.max([h, w])
        im_scale = float(target_size) / float(im_size_min)
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        scales = [float(scale)/target_size*im_scale for scale in TEST_SCALES]
        return scales

    def multi_scale_testing(self):


        flops_total = int(0)
        pbar = tqdm(self.dataset, desc='Multi-scale testing {} on {}'.format(self.model.name, self.dataset.name))
        for img in pbar:
            flops = int(0)

            scales = self._calc_shrink(img.shape[0], img.shape[1])

            for scale in scales:
                flops += self._calc_flops(img, scale, flip=True)

            flops_total += flops
        flops_avg = flops_total / len(self.dataset)
        return flops_avg