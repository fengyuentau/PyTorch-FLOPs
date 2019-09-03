from .module import Module
from .utils import _quadruple
from .. import functional as F
from flops_counter.tensorsize import TensorSize

class _ConstantPadNd(Module):
    __constants__ = ['padding', 'value']

    def __init__(self, value):
        super(_ConstantPadNd, self).__init__()
        self.value = value

    def forward(self, input: TensorSize):
        y = F.pad(input, self.padding, 'constant', self.value)

        return y

    def extra_repr(self):
        return 'padding={}, value={}'.format(self.padding, self.value)

class ConstantPad2d(_ConstantPadNd):
    __constants__ = ['padding', 'value']

    def __init__(self, padding, value):
        super(ConstantPad2d, self).__init__(value)
        self.padding = _quadruple(padding)

class ZeroPad2d(ConstantPad2d):
    def __init__(self, padding):
        super(ZeroPad2d, self).__init__(padding, 0.)