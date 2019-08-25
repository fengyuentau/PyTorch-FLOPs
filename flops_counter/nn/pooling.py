from ._utils import _pair

from .module import Module

class MaxPool2d(Module):
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

    def __repr__(self):
        base = 'MaxPool2d(kernel_size={:s}'.format(str(self.kernel_size))
        if self.stride != (1, 1):
            base += ', stride={:s}'.format(str(self.stride))
        if self.padding != (0, 0):
            base += ', padding={:s}'.format(str(self.padding))
        if self.dilation != (1, 1):
            base += ', dilation={:s}'.format(str(self.dilation))
        if self.return_indices != False:
            base += ', return_indices={:s}'.format(str(self.return_indices))
        if self.ceil_mode != False:
            base += ', ceil_mode={:s}'.format(self.ceil_mode)
        base += ')'
        if self._flops != 0:
            base += ', FLOPs = {:,d}'.format(self._flops)
        return base

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