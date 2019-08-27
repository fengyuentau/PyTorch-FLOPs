from .module import Module

class ReLU(Module):
    def __init__(self,
        inplace=False):

        super(ReLU, self).__init__()

        self.inplace = inplace

    def extra_repr(self):
        parameters = ''
        if self.inplace != False:
            parameters += 'inplace={inplace}'
        return parameters.format(**self.__dict__)

    def _calc_flops(self, x, y):
        cin, hin, win = x
        cout, hout, wout = y
        self._flops = 2 * (cout * hout * wout)

    def forward(self, x):
        '''
        x should be of shape [channels, height, width]
        '''
        assert len(x) == 3, 'input size should be 3, which is [channels, height, width].'

        y = [i for i in x]

        self._calc_flops(x, y)

        return y

class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()


    def _calc_flops(self, x, y):
        cin, hin, win = x
        cout, hout, wout = y
        self._flops = 3 * (cout * hout * wout)

    def forward(self, x):
        '''
        x should be of shape [channels, height, width]
        '''
        assert len(x) == 3, 'input size should be 3, which is [channels, height, width].'

        y = [i for i in x]

        self._calc_flops(x, y)

        return y