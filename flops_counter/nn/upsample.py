import math

from ._utils import _pair

class Upsample(object):
    def __init__(self,
        size=None,
        scale_factor=None,
        mode='nearest',
        align_corners=None):

        assert isinstance(size, tuple) and len(size) == 2, 'Size must be a 2-element tuple.'

        super(Upsample, self).__init__()

        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners

    def _get_flops(self, x, y):
        cin, hin, win = x
        cout, hout, wout = y
        return 13 * abs(cout * hout * wout - cin * hin * win)

    def __call__(self, x):
        def _output(x):
            if self.size is not None:
                return [x[0], self.size[0], self.size[1]]
            self.scale_factor = _pair(self.scale_factor)
            return [x[0], int(math.floor(float(x[0])) * self.scale_factor[0]), int(math.floor(float(x[1])) * self.scale_factor[1])]

        y = _output(x)

        flops = self._get_flops(x, y)

        return y, flops