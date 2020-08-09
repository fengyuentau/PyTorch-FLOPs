import cv2
import numpy as np
from tqdm import tqdm

import flops_counter

class Eval(object):
    def __init__(self, dataset, model, ms=False, max_downsample=1):
        super(Eval, self).__init__()

        self.dataset = dataset
        self.model = model
        self.ms = ms # flag for multi-scale testing or not
        self.max_downsample = max_downsample

    def _calc_shrink(self, h, w):
        raise NotImplementedError

    def _calc_flops(self, img, scale=1, flip=False):
        x = img

        h, w = x.shape[:2]
        scale_h, scale_w = scale, scale
        if self.max_downsample > 1:
            h_new = int(np.ceil(scale * h / self.max_downsample) * self.max_downsample)
            w_new = int(np.ceil(scale * w / self.max_downsample) * self.max_downsample)
            scale_h = h_new / h
            scale_w = w_new / w
            x = cv2.resize(img, None, None, fx=scale_w, fy=scale_h, interpolation=cv2.INTER_LINEAR)
        elif scale != 1:
            x = cv2.resize(img, None, None, fx=scale_w, fy=scale_h, interpolation=cv2.INTER_LINEAR)
        h, w, c = x.shape

        if 'hr' in self.model.name.lower():
            if scale > 1 and (h > 5000 or w > 5000):
                return 0

        x = flops_counter.TensorSize([1, c, h, w])
        y = self.model(x)

        flops = self.model.flops * 2 if flip else self.model.flops

        self.model.set_flops_zero()
        return flops

    def single_scale_testing(self):
        '''Tests the model with scale-1 input. No flip, no shrink calculation.
        '''
        flops_total = int(0)
        for img in tqdm(self.dataset, desc='Scale-1 testing {} on {}'.format(self.model.name, self.dataset.name)):
            flops = self._calc_flops(img)
            flops_total += flops
        return flops_total / len(self.dataset)

    def multi_scale_testing(self):
        '''Tests the model with preset multi scales, possibly with flip and shrink calculation.
        '''
        raise NotImplementedError

    def test(self):
        if self.ms:
            try:
                return self.multi_scale_testing()
            except NotImplementedError as e:
                print('Multi-scale testing for {} is not defined, turning to single-scale testing.'.format(self.model.name))
                return self.single_scale_testing()
        else:
            return self.single_scale_testing()