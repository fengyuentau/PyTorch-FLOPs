import math

from ._utils import _pair

from .module import Module

class Upsample(Module):
    def __init__(self,
        size=None,
        scale_factor=None,
        mode='nearest',
        align_corners=None):

        assert isinstance(size, tuple) and len(size) == 2, 'Size must be a 2-element tuple.'
        assert size or scale_factor, 'Only one of size and scale_factor can be specify.'
        assert mode.lower() == 'bilinear', 'Currently only support Upsample with bilinear mode.'

        super(Upsample, self).__init__()

        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners

    def __repr__(self):
        base = 'Upsample('
        if self.size:
            base += ', size={:s}'.format(str(self.size))
        if self.scale_factor:
            base += ', scale_factor={:s}'.format(str(self.scale_factor))
        if self.mode != 'nearest':
            base += ', mode={:s}'.format(str(self.mode))
        if self.align_corners:
            base += ', align_corners={:s}'.format(str(self.align_corners))
        base += ')'
        return base

    def _calc_flops(self, x, y):
        cin, hin, win = x
        cout, hout, wout = y
        self._flops = 13 * abs(cout * hout * wout - cin * hin * win)

    def forward(self, x):
        def _output(x):
            if self.size is not None:
                return [x[0], self.size[0], self.size[1]]
            self.scale_factor = _pair(self.scale_factor)
            return [x[0], int(math.floor(float(x[0])) * self.scale_factor[0]), int(math.floor(float(x[1])) * self.scale_factor[1])]

        y = _output(x)

        self._calc_flops(x, y)

        return y, flops