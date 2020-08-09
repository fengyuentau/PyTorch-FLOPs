import cv2
import numpy as np
from tqdm import tqdm

from .base import Eval

class SSH(Eval):
    '''Tests SSH
    '''

    def __init__(self, dataset, model, ms=False, max_downsample=1):
        super(SSH, self).__init__(dataset, model, ms, max_downsample)


    def _calc_shrink(self, image_shape, target_size, max_size):
        im_size_min = min(image_shape)
        im_size_max = max(image_shape)

        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if int(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        return im_scale

    def multi_scale_testing(self):
        scales = [500, 800, 1200, 1600, ]
        pyramid_base_size = [800,1200]

        flops_total = int(0)
        pbar = tqdm(self.dataset, desc='Multi-scale testing {} on {}'.format(self.model.name, self.dataset.name))
        for img in pbar:
            flops = int(0)

            base_scale = self._calc_shrink(img.shape[0:2], pyramid_base_size[0], pyramid_base_size[1])
            pyramid_scales = [ float(scale)/pyramid_base_size[0]*base_scale for scale in scales]
            for pyramid_scale in pyramid_scales:
                flops += self._calc_flops(img, pyramid_scale)

            flops_total += flops
        flops_avg = flops_total / len(self.dataset)
        return flops_avg