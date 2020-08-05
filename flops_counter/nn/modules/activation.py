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
        self._flops += 2 * y.nelement

    def forward(self, x: TensorSize):
        assert isinstance(x, TensorSize), \
            'Type of input must be \'{}\'.'.format(TensorSize.__name__)

        y = TensorSize(x._tensor_size)

        return y

class LeakyReLU(Module):
    __constants__ = ['negative_slope', 'inplace']

    def __init__(self, negative_slope=0.01, inplace=False):
        super(LeakyReLU, self).__init__()

        self.negative_slope = negative_slope
        self.inplace = inplace

    def extra_repr(self):
        parameters = ''
        parameters += 'negative_slope={negative_slope}'
        parameters += ', inplace={inplace}'
        return parameters.format(**self.__dict__)

    def _calc_flops(self, x: TensorSize, y: TensorSize):
        bsin = x.value[0]
        bsout = y.value[0]
        assert bsin == bsout, 'Batch size of input and output must be equal'
        self._flops += 6 * y.nelement

    def forward(self, x: TensorSize):
        assert isinstance(x, TensorSize), \
            'Type of input must be \'{}\'.'.format(TensorSize.__name__)

        y = TensorSize(x._tensor_size)

        return y

class PReLU(Module):
    __constants__ = ['num_parameters', 'init']

    def __init__(self, num_parameters=1, init=0.25):
        super(PReLU, self).__init__()

        self.num_parameters = num_parameters
        self.init = init

    def extra_repr(self):
        parameters = ''
        parameters += 'num_parameters={num_parameters}'
        parameters += ', init={init}'
        return parameters.format(**self.__dict__)

    def _calc_flops(self, x: TensorSize, y: TensorSize):
        bsin = x.value[0]
        bsout = y.value[0]
        assert bsin == bsout, 'Batch size of input and output must be equal'
        if self.num_parameters != 1:
            assert self.num_parameters == x.value[1], 'num_parameters must either be equal to 1 or the channels of input tensor.'
        self._flops += 6 * y.nelement

    def forward(self, x: TensorSize):
        assert isinstance(x, TensorSize), \
            'Type of input must be \'{}\'.'.format(TensorSize.__name__)
        if self.num_parameters != 1:
            assert self.num_parameters == x.value[1], 'num_parameters must either be equal to 1 or the channels of input tensor.'

        y = TensorSize(x._tensor_size)

        return y

class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def _calc_flops(self, x: TensorSize, y: TensorSize):
        bsin = x.value[0]
        bsout = y.value[0]
        assert bsin == bsout, 'Batch size of input and output must be equal'
        self._flops += 3 * y.nelement

    def forward(self, x: TensorSize):
        assert isinstance(x, TensorSize), \
            'Type of input must be \'{}\'.'.format(TensorSize.__name__)

        y = TensorSize(x._tensor_size)

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
        self._flops += 3 * y.nelement - ondim * bsout

    def forward(self, x: TensorSize):
        assert isinstance(x, TensorSize), \
            'Type of input must be \'{}\'.'.format(TensorSize.__name__)

        y = TensorSize(x._tensor_size)

        return y