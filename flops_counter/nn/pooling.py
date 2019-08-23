from ._utils import _pair

class MaxPool2d(object):
    def __init__(self,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        return_indices: bool=False,
        ceil_mode: bool=False):

        super(MaxPool2d, self).__init__()

        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def _out(self, i, idx):
        return (i + 2 * self.padding[idx] - self.dilation[idx] * (self.kernel_size[idx] - 1) - 1) // self.stride[idx] + 1

    def _get_flops(self, x, y):
        cin, hin, win = x
        cout, hout, wout = y
        return self.kernel_size[0] * self.kernel_size[1] * cout * hout * wout

    def __call__(self, x):
        '''
        x should be of shape [channels, height, width]
        '''
        assert len(x) == 3, 'input size should be 3, which is [channels, height, width].'

        cin, hin, win = x
        hout = self._out(hin, 0)
        wout = self._out(win, 1)
        y = [cin, hout, wout]

        flops = self._get_flops(x, y)

        return y, flops