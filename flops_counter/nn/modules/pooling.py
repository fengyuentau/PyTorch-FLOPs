from .utils import _pair

from .module import Module

class MaxPool2d(Module):
    __constants__ = ['kernel_size', 'stride', 'padding', 'dilation',
                     'return_indices', 'ceil_mode']

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

    def extra_repr(self):
        parameters = 'kernel_size={kernel_size}'
        if self.stride != (1, 1):
            parameters += ', stride={stride}'
        if self.padding != (0, 0):
            parameters += ', padding={padding}'
        if self.dilation != (1, 1):
            parameters += ', dilation={dilation}'
        if self.return_indices != False:
            parameters += ', return_indices={return_indices}'
        if self.ceil_mode != False:
            parameters += ', ceil_mode={ceil_mode}'
        return parameters.format(**self.__dict__)

    def _calc_out(self, i, idx):
        return (i + 2 * self.padding[idx] - self.dilation[idx] * (self.kernel_size[idx] - 1) - 1) // self.stride[idx] + 1

    def _calc_flops(self, x, y):
        cin, hin, win = x
        cout, hout, wout = y
        self._flops = self.kernel_size[0] * self.kernel_size[1] * cout * hout * wout

    def forward(self, x):
        '''
        x should be of shape [channels, height, width]
        '''
        assert len(x) == 3, 'input size should be 3, which is [channels, height, width].'

        cin, hin, win = x
        hout = self._calc_out(hin, 0)
        wout = self._calc_out(win, 1)
        y = [cin, hout, wout]

        self._calc_flops(x, y)

        return y

class AdaptiveAvgPool2d(Module):
    __constants__ = ['output_size']
    def __init__(self, output_size):
        super(AdaptiveAvgPool2d, self).__init__()
        self.output_size = _pair(output_size)

    def extra_repr(self):
        return 'output_size={output_size}'.format(**self.__dict__)

    def _calc_out(self, i, idx):
        return self.output_size[idx]

    def _calc_flops(self, x, y):
        raise NotImplementedError

    def forward(self, x):
        '''
        x should be of shape [channels, height, width]
        '''
        assert len(x) == 3, 'input size should be 3, which is [channels, height, width].'

        cin, hin, win = x
        hout = self._calc_out(hin, 0)
        wout = self._calc_out(win, 1)
        y = [cin, hout, wout]

        # self._calc_flops(x, y)

        return y