from functools import reduce

from .module import Module

from flops_counter.tensorsize import TensorSize

class ReLU(Module):
    __constants__ = ['inplace']

    def __init__(self,
        inplace=False):

        super(ReLU, self).__init__()

        self.inplace = inplace

    def extra_repr(self):
        parameters = ''
        parameters += 'inplace={inplace}'
        return parameters.format(**self.__dict__)

    def _calc_flops(self, x: TensorSize, y: TensorSize):
        bsin = x.value[0]
        bsout = y.value[0]
        assert bsin == bsout, 'Batch size of input and output must be equal'
        self._flops = 2 * y.nelement

    def forward(self, x: TensorSize):
        assert isinstance(x, TensorSize), \
            'Type of input must be \'{}\'.'.format(TensorSize.__name__)

        y = TensorSize(x._tensor_size)

        self._input = x
        self._output = y

        return y

class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def _calc_flops(self, x: TensorSize, y: TensorSize):
        bsin = x.value[0]
        bsout = y.value[0]
        assert bsin == bsout, 'Batch size of input and output must be equal'
        self._flops = 3 * y.nelement

    def forward(self, x: TensorSize):
        assert isinstance(x, TensorSize), \
            'Type of input must be \'{}\'.'.format(TensorSize.__name__)

        y = TensorSize(x._tensor_size)

        self._input = x
        self._output = y

        return y


class Softmax(Module):
    __constants__ = ['dim']

    def __init__(self, dim=None):
        super(Softmax, self).__init__()
        self.dim = dim

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    def extra_repr(self):
        return 'dim={dim}'.format(dim=self.dim)

    def _calc_flops(self, x: TensorSize, y: TensorSize):
        bsin = x.value[0]
        bsout = y.value[0]
        assert bsin == bsout, 'Batch size of input and output must be equal'

        ondim = y.value[self.dim]
        # flops_softmax = bsout * ((2 * h * w - 1) + h * w) * c = bsout * (3hwc - c)
        self._flops = 3 * y.nelement - ondim * bsout

    def forward(self, x: TensorSize):
        assert isinstance(x, TensorSize), \
            'Type of input must be \'{}\'.'.format(TensorSize.__name__)

        y = TensorSize(x._tensor_size)

        self._input = x
        self._output = y

        return y