class ReLU(object):
    def __init__(self,
        inplace=False):

        super(ReLU, self).__init__()

        self.inplace = inplace

    def _get_flops(self, x, y):
        cin, hin, win = x
        cout, hout, wout = y
        return 2 * (cout * hout * wout)

    def __call__(self, x):
        '''
        x should be of shape [channels, height, width]
        '''
        assert len(x) == 3, 'input size should be 3, which is [channels, height, width].'

        y = [i for i in x]

        flops = self._get_flops(x, y)

        return y, flops