from .module import Module

class BatchNorm2d(Module):
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

    def __repr__(self):
        base = 'BatchNorm2d(num_features={:d}'.format(self.num_features)
        if self.eps != 1e-05:
            base += ', eps={:e}'.format(self.eps)
        if self.momentum != 0.1:
            base += ', momentum={:f}'.format(self.momentum)
        if self.affine != True:
            base += ', affine={:s}'.format(str(self.affine))
        if self.track_running_stats != True:
            base += ', track_running_stats={:s}'.format(str(self.track_running_stats))
        base += ')'
        if self._flops != 0:
            base += ', FLOPs = {:,d}'.format(self._flops)
        return base

    def _calc_flops(self, x, y):
        cin, hin, win = x
        cout, hout, wout = y
        self._flops = 6 * (cout * hout * wout)

    def forward(self, x):
        '''
        x should be of shape [channels, height, width]
        '''
        assert len(x) == 3, 'input size should be 3, which is [channels, height, width].'
        assert x[0] == self.num_features, 'The channel of input {:d} does not match with the definition {:d}'.format(x[0], self.in_channels)

        y = [i for i in x]

        self._calc_flops(x, y)

        return y

class L2Norm2d(Module):
    def __init__(self,
        num_features: int,
        gamma_init: int=20):

        super(L2Norm2d, self).__init__()

        self.num_features = num_features
        self.gamma_init = gamma_init

    def __repr__(self):
        base = 'L2Norm2d(num_features={:d}'.format(self.num_features)
        if self.gamma_init != 20:
            base += ', gamma_init={:d}'.format(self.gamma_init)
        base += ')'
        if self._flops != 0:
            base += ', FLOPs = {:,d}'.format(self._flops)
        return base

    def _calc_flops(self, x, y):
        cin, hin, win = x
        cout, hout, wout = y
        self._flops = 3 * (cout * hout * wout)

    def forward(self, x):
        '''
        x should be of shape [channels, height, width]
        '''
        assert len(x) == 3, 'input size should be 3, which is [channels, height, width].'
        assert x[0] == self.num_features, 'The channel of input {:d} does not match with the definition {:d}'.format(x[0], self.in_channels)

        y = [i for i in x]

        self._calc_flops(x, y)

        return y