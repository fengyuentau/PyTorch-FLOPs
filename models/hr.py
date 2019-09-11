import flops_counter
import flops_counter.nn as nn
import flops_counter.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=False, add_relu=True, add_bn=True, eps=1e-5):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=bias)
        self.relu = None
        self.bn = None

        if add_relu:
            self.relu = nn.ReLU()
        if add_bn:
            self.bn = nn.BatchNorm2d(out_channel, eps=eps)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out) if self.bn else out
        out = self.relu(out) if self.relu else out
        return out

class ConvTransLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=False):
        super(ConvTransLayer, self).__init__()
        self.transconv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride, padding, bias=bias)

    def forward(self, x):
        return self.transconv(x)

class ResidualBlock(nn.Module):
    def __init__(self, name, in_channel, mid_channel, out_channel, downsample=False):
        super(ResidualBlock, self).__init__()

        stride = 2 if name.startswith('res3a') or name.startswith('res4a') else 1
        self.cb1 = ConvBlock(in_channel, mid_channel, 1, stride, 0, add_relu=True)
        self.cb2 = ConvBlock(mid_channel, mid_channel, 3, padding=1, add_relu=True)
        self.cb3 = ConvBlock(mid_channel, out_channel, 1, padding=0, add_relu=False)

        self.branch = None
        if downsample:
            self.branch = ConvBlock(in_channel, out_channel, 1, stride, 0, add_relu=False)

        self.eltadd = nn.EltAdd()

    def forward(self, x):
        out = self.cb1(x)
        out = self.cb2(out)
        out = self.cb3(out)

        if self.branch:
            branch = self.branch(x)
            return self.eltadd(out, branch)
        return self.eltadd(out, x)




class HR(nn.Module):
    def __init__(self):
        super(HR, self).__init__()

        self.conv = ConvBlock(3, 64, 7, 2, add_relu=True)
        self.pool = nn.MaxPool2d(3, 2, 1)

        self.res2a = ResidualBlock('res2a', 64, 64, 256, downsample=True)
        self.res2b = ResidualBlock('res2b', 256, 64, 256)
        self.res2c = ResidualBlock('res2c', 256, 64, 256)

        self.res3a = ResidualBlock('res3a', 256, 128, 512, downsample=True)
        self.res3b1 = ResidualBlock('res3b1', 512, 128, 512)
        self.res3b2 = ResidualBlock('res3b2', 512, 128, 512)
        self.res3b3 = ResidualBlock('res3b3', 512, 128, 512)

        self.res4a = ResidualBlock('res4a', 512, 256, 1024, downsample=True)
        self.res4bX = nn.ModuleList()
        for i in range(0, 22):
            self.res4bX.append(ResidualBlock('res4bX', 1024, 256, 1024))

        # Detection Head
        self.score_res4 = ConvBlock(1024, 125, 1, bias=True, add_relu=False, add_bn=False)
        self.score4 = ConvTransLayer(125, 125, 4, 2, 0)

        self.score_res3 = ConvBlock(512, 125, 1, bias=True, add_relu=False, add_bn=False)

        self.eltadd = nn.EltAdd()

    @property
    def name(self):
        return self._get_name() + '_ResNet101'

    def forward(self, x):
        conv = self.conv(x)
        pool = self.pool(conv)

        res2a = self.res2a(pool)
        res2b = self.res2b(res2a)
        res2c = self.res2c(res2b)

        res3a = self.res3a(res2c)
        res3b1 = self.res3b1(res3a)
        res3b2 = self.res3b2(res3b1)
        res3b3 = self.res3b3(res3b2)

        res4a = self.res4a(res3b3)
        res4bX = res4a
        for i in range(0, 22):
            res4bX = self.res4bX[i](res4bX)
        self.res4bX.settle(res4a, res4bX)

        # Detection Head
        score_res4 = self.score_res4(res4bX)
        score4 = self.score4(score_res4)

        score_res3 = self.score_res3(res3b3)

        if score4.value[2] != score_res3.value[2] or score4.value[3] != score_res3.value[3]:
            pads = (0, score_res3.value[3] - score4.value[3], 0, score_res3.value[2] - score4.value[2])
            score4 = F.pad(score4, pads)


        score_final = self.eltadd(score4, score_res3)
        return score_final