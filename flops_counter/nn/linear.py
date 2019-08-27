from .module import Module

# https://pytorch.org/docs/stable/nn.html#linear
class Linear(Module):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

    def extra_repr(self):
        parameters = 'in_features={in_features}, out_features={out_features}'
        if self.bias != True:
            parameters += ', bias={bias}'
        return parameters.format(**self.__dict__)

    def _calc_flops(self, x, y):
        cin, hin, win = x
        cout, hout, wout = y
        self._flops = (2 * self.in_features - (0 if self.bias else 1)) * self.out_features

    def forward(self, x):
        '''
        x should be of shape [channels, height, width]
        '''
        assert x[-1] == self.in_features, 'last dimension {:d} does not match with in_features {:d}.'.format(x[-1], self.in_features)

        y = [_x for _x in x]
        y[-1] = self.out_features

        self._calc_flops(x, y)

        return y