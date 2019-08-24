from .module import Module

class EltAdd(Module):
    def __init__(self):
        super(EltAdd, self).__init__()

    def __repr__(self):
        base = 'EltAdd('
        base += ')'
        return base

    def _calc_flops(self, x, y):
        cin, hin, win = x
        cout, hout, wout = y
        self._flops = cout * hout * wout

    def forward(self, a, b):
        assert len(a) == len(b), 'Dimension of a and b must be equal.'
        assert a == b, 'Size of a and b must be equal.'

        x = [i for i in a]
        y = [i for i in a]

        self._calc_flops(x, y)

        return y

class EltMul(Module):
    def __init__(self):
        super(EltMul, self).__init__()

    def __repr__(self):
        base = 'EltMul('
        base += ')'
        return base

    def _calc_flops(self, x, y):
        cin, hin, win = x
        cout, hout, wout = y
        self._flops = cout * hout * wout

    def forward(self, a, b):
        assert len(a) == len(b), 'Dimension of a and b must be equal.'
        assert a == b, 'Size of a and b must be equal.'

        x = [i for i in a]
        y = [i for i in a]

        self._calc_flops(x, y)

        return y