import math

from .utils import _pair

from .module import Module
from flops_counter.tensorsize import TensorSize

class Upsample(Module):
    def __init__(self,
        size=None,
        scale_factor=None,
        mode='nearest',
        align_corners=None):

        assert (size is not None and scale_factor is None) or (size is None and scale_factor is not None), \
            'Only one of size and scale_factor can be specify.'
        if size is not None:
            assert isinstance(size, tuple) and len(size) == 2, 'Size must be a 2-element tuple.'
        assert mode.lower() == 'bilinear', 'Currently only support Upsample with bilinear mode.'

        super(Upsample, self).__init__()

        self.size = size
        # if isinstance(scale_factor, tuple):
        #     self.scale_factor = tuple(float(factor) for factor in scale_factor)
        # else:
        #     self.scale_factor = float(scale_factor) if scale_factor else None
        self.scale_factor = _pair(scale_factor)
        self.mode = mode
        self.align_corners = align_corners

    def extra_repr(self):
        parameters = ''
        if self.size is not None:
            parameters += 'size={size}, '
        if self.scale_factor is not None:
            parameters += 'scale_factor={scale_factor}, '
        if self.mode != 'nearest':
            parameters += 'mode={mode}'
        if self.align_corners is not None:
            parameters += ', align_corners={align_corners}'
        return parameters.format(**self.__dict__)

    def _calc_flops(self, x: TensorSize, y: TensorSize):
        bsin, cin, hin, win = x.value
        bsout, cout, hout, wout = y.value
        assert bsin == bsout, 'Batch size of input and output must be equal'
        self._flops = 13 * abs(y.nelement - x.nelement)

    def forward(self, x: TensorSize):
        assert isinstance(x, TensorSize), \
            'Type of input must be \'{}\'.'.format(TensorSize.__name__)

        if x.dim == 4:
            bsin, cin, hin, win = x.value
            _out = [bsin, cin, hin, win]
            if self.size is not None:
                _out = [bsin, cin, self.size[0], self.size[1]]
            else:
                _out = [bsin, cin, hin * self.scale_factor[0], win * self.scale_factor[1]]
            y = TensorSize(_out)

            self._input = x
            self._output = y

            return y
        else:
            raise NotImplementedError('Not implemented yet for \'{:s}\' with dimension {:d} != 4.'.format(TensorSize.__name__, x.dim))