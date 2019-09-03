from .module import Module
from flops_counter.tensorsize import TensorSize

class BatchNorm2d(Module):
    __constants__ = ['track_running_stats', 'momentum', 'eps',
                     'num_features', 'affine']

    def __init__(self,
        num_features: int,
        eps: float=1e-05,
        momentum: float=0.1,
        affine: bool=True,
        track_running_stats: bool=True):

        super(BatchNorm2d, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

    def extra_repr(self):
        parameters = 'num_features={num_features}'
        if self.eps != 1e-05:
            parameters += ', eps={eps}'
        if self.momentum != 0.1:
            parameters += ', momentum={momentum}'
        if self.affine != True:
            parameters += ', affine={affine}'
        if self.track_running_stats != True:
            parameters += ', track_running_stats={track_running_stats}'
        return parameters.format(**self.__dict__)

    def _calc_flops(self, x: TensorSize, y: TensorSize):
        bsin, cin, hin, win = x.value
        bsout, cout, hout, wout = y.value
        assert bsin == bsout, 'Batch size of input and output must be equal'
        self._flops = 6 * y.nelement

    def forward(self, x: TensorSize):
        assert isinstance(x, TensorSize), \
            'Type of input must be \'{}\'.'.format(TensorSize.__name__)
        assert x.value[1] == self.num_features, 'The channel of input {:d} does not match with the definition {:d}'.format(x.value[0], self.num_features)

        if x.dim == 4:
            y = TensorSize(x.value)

            self._input = x
            self._output = y

            return y
        else:
            raise NotImplementedError('Not implemented yet for \'{:s}\' with dimension {:d} != 4.'.format(TensorSize.__name__, x.dim))

class L2Norm2d(Module):
    __constants__ = ['num_features', 'gamma_init']

    def __init__(self,
        num_features: int,
        gamma_init: int=20):

        super(L2Norm2d, self).__init__()

        self.num_features = num_features
        self.gamma_init = gamma_init

    def extra_repr(self):
        parameters = 'num_features={num_features}'
        if self.gamma_init != 20:
            parameters += ', gamma_init={gamma_init}'
        return parameters.format(**self.__dict__)

    def _calc_flops(self, x: TensorSize, y: TensorSize):
        bsin, cin, hin, win = x.value
        bsout, cout, hout, wout = y.value
        assert bsin == bsout, 'Batch size of input and output must be equal'
        self._flops = 3 * y.nelement

    def forward(self, x: TensorSize):
        assert isinstance(x, TensorSize), \
            'Type of input must be \'{}\'.'.format(TensorSize.__name__)
        assert x.value[1] == self.num_features, 'The channel of input {:d} does not match with the definition {:d}'.format(x.value[0], self.num_features)

        if x.dim == 4:
            y = TensorSize(x.value)

            self._input = x
            self._output = y

            return y
        else:
            raise NotImplementedError('Not implemented yet for \'{:s}\' with dimension {:d} != 4.'.format(TensorSize.__name__, x.dim))