from ._utils import _pair

from .module import Module

class Conv2d(Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias: bool=True,
        padding_mode='zeros'):

        super(Conv2d, self).__init__()

        assert in_channels % groups == 0 and out_channels % groups == 0, 'in_channels and out_channels must be dividable by groups.'

        self.in_channels    = in_channels
        self.out_channels   = out_channels
        self.kernel_size    = _pair(kernel_size)
        self.stride         = _pair(stride)
        self.padding        = _pair(padding)
        self.dilation       = _pair(dilation)
        self.groups         = groups
        self.bias           = bias
        self.padding_mode   = padding_mode

    def __repr__(self):
        base = 'Conv2d({:d}, {:d}, kernel_size={:s}'.format(self.in_channels, self.out_channels, str(self.kernel_size))
        if self.stride != (1, 1):
            base += ', stride={:s}'.format(str(self.stride))
        if self.padding != (0, 0):
            base += ', padding={:s}'.format(str(self.padding))
        if self.dilation != (1, 1):
            base += ', dilation={:s}'.format(str(self.dilation))
        if self.groups != 1:
            base += ', groups={:s}'.format(self.groups)
        if self.bias != True:
            base += ', bias={:s}'.format(str(self.bias))
        if self.padding_mode != 'zeros':
            base += ', padding_mode={:s}'.format(self.padding_mode)
        base += ')'
        if self._flops != 0:
            base += ', FLOPs = {:,d}'.format(self._flops)
        return base

    def _calc_out(self, i, idx):
        return (i + 2 * self.padding[idx] - self.dilation[idx] * (self.kernel_size[idx] - 1) - 1) // self.stride[idx] + 1

    def _calc_flops(self, x, y):
        cin, hin, win = x
        cout, hout, wout = y
        self._flops = (2 * cin * self.kernel_size[0] * self.kernel_size[1] - (0 if self.bias else 1)) * (cout // self.groups) * hout * wout

    def forward(self, x):
        '''
        x should be of shape [channels, height, width]
        '''
        assert len(x) == 3, 'input size should be 3, which is [channels, height, width].'
        assert x[0] == self.in_channels, 'The channel of input {:d} does not match with the definition {:d}'.format(x[0], self.in_channels)

        cin, hin, win = x
        hout = self._calc_out(hin, 0)
        wout = self._calc_out(win, 1)
        y = [self.out_channels, hout, wout]

        self._calc_flops(x, y)

        return y


class ConvTranspose2d(Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias: bool=True,
        dilation=1,
        padding_mode='zeros'):

        super(ConvTranspose2d, self).__init__()

        self.in_channels    = in_channels
        self.out_channels   = out_channels
        self.kernel_size    = _pair(kernel_size)
        self.stride         = _pair(stride)
        self.padding        = _pair(padding)
        self.output_padding = _pair(output_padding)
        self.groups         = groups
        self.bias           = bias
        self.dilation       = _pair(dilation)
        self.padding_mode   = padding_mode

    def __repr__(self):
        base = 'ConvTranspose2d({:d}, {:d}, kernel_size={:s}'.format(self.in_channels, self.out_channels, str(self.kernel_size))
        if self.stride != (1, 1):
            base += ', stride={:s}'.format(str(self.stride))
        if self.padding != (0, 0):
            base += ', padding={:s}'.format(str(self.padding))
        if self.output_padding != (0, 0):
            base += ', output_padding={:s}'.format(str(self.output_padding))
        if self.groups != 1:
            base += ', groups={:s}'.format(self.groups)
        if self.bias != True:
            base += ', bias={:s}'.format(str(self.bias))
        if self.dilation != (1, 1):
            base += ', dilation={:s}'.format(str(self.dilation))
        if self.padding_mode != 'zeros':
            base += ', padding_mode={:s}'.format(self.padding_mode)
        base += ')'
        if self._flops != 0:
            base += ', FLOPs = {:,d}'.format(self._flops)
        return base

    def _calc_out(self, i, idx):
        return (i - 1) * self.stride[idx] - 2 * self.padding[idx] + self.dilation[idx] * (self.kernel_size[idx] - 1) + self.output_padding[idx] + 1

    def _calc_flops(self, x, y):
        cin, hin, win = x
        cout, hout, wout = y
        self._flops = (2 * cin * self.kernel_size[0] * self.kernel_size[1] - (0 if self.bias else 1)) * (cout // self.groups) * (hout * wout - self.output_padding[0] * self.output_padding[1])

    def forward(self, x):
        '''
        x should be of shape [channels, height, width]
        '''
        assert len(x) == 3, 'input size should be 3, which is [channels, height, width].'
        assert x[0] == self.in_channels, 'The channel of input {:d} does not match with the definition {:d}'.format(x[0], self.in_channels)

        cin, hin, win = x
        hout = self._calc_out(hin, 0)
        wout = self._calc_out(win, 1)
        y = [self.out_channels, hout, wout]

        self._calc_flops(x, y)

        return y