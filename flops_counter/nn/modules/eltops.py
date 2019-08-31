from .module import Module
from flops_counter.tensorsize import TensorSize

class EltAdd(Module):
    def __init__(self):
        super(EltAdd, self).__init__()

    def _calc_flops_Nd(self, x: TensorSize, y: TensorSize):
        bsin = x.value[0]
        bsout = y.value[0]
        assert bsin == bsout, 'Batch size of input and output must be equal'
        self._flops = y.nelement

    def forward(self, a: TensorSize, b: TensorSize=None):
        assert isinstance(a, TensorSize), \
            'Type of input must be \'{}\'.'.format(TensorSize.__name__)
        if b is not None:
            assert isinstance(b, TensorSize), \
                'Type of input must be \'{}\'.'.format(TensorSize.__name__)
            assert a.dim == b.dim, 'Dimension of a and b must be equal.'
            assert a.value == b.value, 'Size of a and b must be equal.'

        x = TensorSize(a.value)
        y = TensorSize(a.value)

        self._calc_flops_Nd(x, y)

        self._input = x
        self._output = y

        return y

class EltMul(Module):
    def __init__(self):
        super(EltMul, self).__init__()

    def _calc_flops_Nd(self, x: TensorSize, y: TensorSize):
        bsin = x.value[0]
        bsout = y.value[0]
        assert bsin == bsout, 'Batch size of input and output must be equal'
        self._flops = y.nelement

    def forward(self, a: TensorSize, b: TensorSize=None):
        assert isinstance(a, TensorSize), \
            'Type of input must be \'{}\'.'.format(TensorSize.__name__)
        if b is not None:
            assert isinstance(b, TensorSize), \
                'Type of input must be \'{}\'.'.format(TensorSize.__name__)
            assert a.dim == b.dim, 'Dimension of a and b must be equal.'
            assert a.value == b.value, 'Size of a and b must be equal.'

        x = TensorSize(a.value)
        y = TensorSize(a.value)

        self._calc_flops_Nd(x, y)

        self._input = x
        self._output = y

        return y