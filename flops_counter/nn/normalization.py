class BatchNorm2d(object):
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

    def _get_flops(self, x, y):
        cin, hin, win = x
        cout, hout, wout = y
        return 6 * (cout * hout * wout)

    def __call__(self, x):
        '''
        x should be of shape [channels, height, width]
        '''
        assert len(x) == 3, 'input size should be 3, which is [channels, height, width].'
        assert x[0] == self.num_features, 'The channel of input {:d} does not match with the definition {:d}'.format(x[0], self.in_channels)

        y = [i for i in x]

        flops = self._get_flops(x, y)

        return y, flops

class L2Norm2d(object):
    def __init__(self,
        num_features: int,
        gamma_init: int=20):

        super(L2Norm2d, self).__init__()

        self.num_features = num_features
        self.gamma_init = gamma_init

    def _get_flops(self, x, y):
        cin, hin, win = x
        cout, hout, wout = y
        return 3 * (cout * hout * wout)

    def __call__(self, x):
        '''
        x should be of shape [channels, height, width]
        '''
        assert len(x) == 3, 'input size should be 3, which is [channels, height, width].'
        assert x[0] == self.num_features, 'The channel of input {:d} does not match with the definition {:d}'.format(x[0], self.in_channels)

        y = [i for i in x]

        flops = self._get_flops(x, y)

        return y, flops