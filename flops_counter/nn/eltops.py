class EltAdd(object):
    def __init__(self):
        super(EltAdd, self).__init__()

    def _get_flops(self, x, y):
        cin, hin, win = x
        cout, hout, wout = y
        return cout * hout * wout

    def __call__(self, a, b):
        assert len(a) == len(b), 'Dimension of a and b must be equal.'
        assert a == b, 'Size of a and b must be equal.'

        y = a

        flops = self._get_flops(a, y)

        return y, flops

class EltMul(object):
    def __init__(self):
        super(EltMul, self).__init__()

    def _get_flops(self, x, y):
        cin, hin, win = x
        cout, hout, wout = y
        return cout * hout * wout

    def __call__(self, a, b):
        assert len(a) == len(b), 'Dimension of a and b must be equal.'
        assert a == b, 'Size of a and b must be equal.'

        y = a

        flops = self._get_flops(a, y)

        return y, flops