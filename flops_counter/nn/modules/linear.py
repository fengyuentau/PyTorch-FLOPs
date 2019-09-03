from .module import Module
from flops_counter.tensorsize import TensorSize

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
        bsin = x.value[0]
        bsout = y.value[0]
        assert bsin == bsout, 'Batch size of input and output must be equal'
        self._flops += (2 * self.in_features - (0 if self.bias else 1)) * self.out_features * bsout

    def forward(self, x: TensorSize):
        assert isinstance(x, TensorSize), \
            'Type of input must be \'{}\'.'.format(TensorSize.__name__)
        assert x.value[-1] == self.in_features, 'last dimension {:d} does not match with in_features {:d}.'.format(x.value[-1], self.in_features)

        y = [i for i in x.value]
        y[-1] = self.out_features
        y = TensorSize(y)

        return y