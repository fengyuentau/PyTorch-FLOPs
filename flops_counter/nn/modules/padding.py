from .module import Module
from .utils import _quadruple

class _ConstantPadNd(Module):
    __constants__ = ['padding', 'value']

    def __init__(self, value):
        super(_ConstantPadNd, self).__init__()
        self.value = value

    def forward(self, input):
        # return F.pad(input, self.padding, 'constant', self.value)
        assert len(self.padding) % 2 == 0, 'Padding length must be divisible by 2'
        assert len(self.padding) // 2 <= len(input), 'Padding length too large'

        if len(self.padding) == 4 and len(input) == 3:
            cin, hin, win = input
            pleft, pright, ptop, pbottom = self.padding

            cout = cin
            hout = hin + ptop + pbottom
            wout = win + pleft + pright
            y = [cout, hout, wout]
            return y
        else:
            raise NotImplementedError

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