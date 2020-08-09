import flops_counter
import flops_counter.nn as nn
import flops_counter.nn.functional as F

class EXTD(nn.Module):
    def __init__(self, model_type='32'):
        super(EXTD, self).__init__()
        self.model_type = int(model_type)

        self.mobilenet = MobileNetV2(model_type=self.model_type)
        if self.model_type == 32:
            self.base = nn.ModuleList(self.mobilenet.features)[:8]
        elif self.model_type == 48 or self.model_type == 64:
            self.base = nn.ModuleList(self.mobilenet.features)[:6]
        else:
            raise NotImplementedError

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.eltadd = nn.EltAdd()
        self.upfeat = []
        for it in range(5):
            self.upfeat.append(upsample(in_channels=self.model_type, out_channels=self.model_type))
        self.upfeat = nn.ModuleList(self.upfeat)

        self.loc = []
        self.conf = []
        self.net_source = [4, 4, 4, 4]
        self.feature_dim = []
        if self.model_type == 32:
            self.feature_dim += [self.base[4].conv[-3].out_channels]
            for idx in self.net_source:
                self.feature_dim += [self.base[idx].conv[-3].out_channels]
        else:
            self.feature_dim += [self.base[4].conv[-2].out_channels]
            for idx in self.net_source:
                self.feature_dim += [self.base[idx].conv[-2].out_channels]

        self.loc += [nn.Conv2d(self.feature_dim[0], 4, kernel_size=3, padding=1)]
        self.conf += [nn.Conv2d(self.feature_dim[0], 4, kernel_size=3, padding=1)]
        for k, v in enumerate(self.net_source, 1):
            self.loc += [nn.Conv2d(self.feature_dim[k], 4, kernel_size=3, padding=1)]
            self.conf += [nn.Conv2d(self.feature_dim[k], 2, kernel_size=3, padding=1)]
        self.loc += [nn.Conv2d(self.model_type, 4, kernel_size=3, padding=1)]
        self.conf += [nn.Conv2d(self.model_type, 2, kernel_size=3, padding=1)]
        self.loc = nn.ModuleList(self.loc)
        self.conf = nn.ModuleList(self.conf)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        sources = list()
        loc = list()
        conf = list()

        end = 8 if self.model_type == 32 else 6
        for k in range(end):
            x = self.base[k](x)
        s1 = x  # (640, 640) -> (320, 320)

        for k in range(2, end):
            x = self.base[k](x)
        s2 = x  # (160, 160) -> (80, 80)

        for k in range(2, end):
            x = self.base[k](x)
        s3 = x  # (80, 80) -> (40, 40)

        for k in range(2, end):
            x = self.base[k](x)
        s4 = x  # (40, 40) -> (20, 20)

        for k in range(2, end):
            x = self.base[k](x)
        s5 = x  # (20, 20) -> (10, 10)

        for k in range(2, end):
            x = self.base[k](x)
        s6 = x  # (10, 10) -> (5, 5)

        sources.append(s6)

        # def upsample_add(seq, source, target, up_handle, add_handle):
        u1 = upsample_add(self.upfeat[0], s6, s5, self.upsample, self.eltadd)
        sources.append(u1)
        u2 = upsample_add(self.upfeat[0], u1, s4, self.upsample, self.eltadd)
        sources.append(u2)
        u3 = upsample_add(self.upfeat[0], u2, s3, self.upsample, self.eltadd)
        sources.append(u3)
        u4 = upsample_add(self.upfeat[0], u3, s2, self.upsample, self.eltadd)
        sources.append(u4)
        u5 = upsample_add(self.upfeat[0], u4, s1, self.upsample, self.eltadd)
        sources.append(u5)
        sources = sources[::-1]

        loc_x = self.loc[0](sources[0])
        conf_x = self.conf[0](sources[0])
        conf_x_b, conf_x_c, conf_x_h, conf_x_w = conf_x.value
        conf_x = flops_counter.TensorSize([conf_x_b, 2, conf_x_h, conf_x_w])

        loc.append(loc_x.permute(0, 2, 3, 1))
        conf.append(conf_x.permute(0, 2, 3, 1))

        for i in range(1, len(sources)):
            x = sources[i]
            conf.append(self.conf[i](x).permute(0, 2, 3, 1))
            loc.append(self.loc[i](x).permute(0, 2, 3, 1))

        loc = flops_counter.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = flops_counter.cat([o.view(o.size(0), -1) for o in conf], 1)

        return loc.view(loc.size(0), -1, 4), self.softmax(conf.view(conf.size(0), -1, 2))


    @property
    def name(self):
        return self._get_name() + '-' + str(self.model_type)

def conv_bn(inp, oup, stride, k_size=3):
    return nn.Sequential(
        nn.Conv2d(inp, oup, k_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.PReLU()
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.PReLU()
    )

def upsample(in_channels, out_channels): # should use F.inpterpolate
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3),
                  stride=1, padding=1, groups=in_channels, bias=False),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
                  bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

def upsample_add(seq, source, target, up_handle, add_handle):
    up = up_handle(source)
    if up.value[2] != target.value[2] or up.value[3] != target.value[3]:
        pad = (0, target.value[3] - up.value[3], 0, target.value[2] - up.value[2])
        up = F.pad(up, pad)
    dst = add_handle(seq(up), target)
    return dst

class gated_conv1x1(nn.Module):
    def __init__(self, inc=128, outc=128):
        super(gated_conv1x1, self).__init__()
        self.inp = int(inc/2)
        self.oup = int(outc/2)
        self.conv1x1_1 = nn.Conv2d(self.inp, self.oup, 1, 1, 0, bias=False)
        self.gate_1 = nn.Conv2d(self.inp, self.oup, 1, 1, 0, bias=True)
        self.sigmoid1 = nn.Sigmoid()
        self.eltmul1 = nn.EltMul()

        self.conv1x1_2 = nn.Conv2d(self.inp, self.oup, 1, 1, 0, bias=False)
        self.gate_2 = nn.Conv2d(self.inp, self.oup, 1, 1, 0, bias=True)
        self.sigmoid2 = nn.Sigmoid()
        self.eltmul2 = nn.EltMul()

    def forward(self, x):
        # x_1 = x[:, :self.inp, :, :]
        # x_2 = x[:, self.inp:, :, :]
        b, c, h, w = x.value
        x_1 = flops_counter.TensorSize([b, self.inp, h, w])
        x_2 = flops_counter.TensorSize([b, c-self.inp, h, w])

        a_1 = self.conv1x1_1(x_1)
        # g_1 = F.sigmoid(self.gate_1(x_1))
        g_1 = self.sigmoid1(self.gate_1(x_1))

        a_2 = self.conv1x1_2(x_2)
        # g_2 = F.sigmoid(self.gate_2(x_2))
        g_2 = self.sigmoid2(self.gate_2(x_2))

        ret = flops_counter.cat((self.eltmul1(a_1, g_1), self.eltmul2(a_2, g_2)), 1)

        return ret

class InvertedResidual_dwc(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, model_type=32):
        super(InvertedResidual_dwc, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = []

        if expand_ratio == 1:
            self.conv.append(nn.Conv2d(inp, hidden_dim, kernel_size=(3, 3), stride=stride, padding=1, groups=hidden_dim))
            self.conv.append(nn.BatchNorm2d(hidden_dim))
            self.conv.append(nn.PReLU())
            self.conv.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False))
            self.conv.append(nn.BatchNorm2d(oup))
            if model_type == 32:
                self.conv.append(nn.PReLU())
        else:
            self.conv.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
            self.conv.append(nn.BatchNorm2d(hidden_dim))
            self.conv.append(nn.PReLU())
            self.conv.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 3), stride=stride, padding=1, groups=hidden_dim))
            self.conv.append(nn.BatchNorm2d(hidden_dim))
            self.conv.append(nn.PReLU())
            self.conv.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False))
            self.conv.append(nn.BatchNorm2d(oup))
            if model_type == 32:
                self.conv.append(nn.PReLU())

        self.conv = nn.Sequential(*self.conv)
        if self.use_res_connect:
            self.eltadd = nn.EltAdd()

    def forward(self, x):
        if self.use_res_connect:
            return self.eltadd(x, self.conv(x))
        else:
            return self.conv(x)

class MobileNetV2(nn.Module): #mobileNet v2
    def __init__(self, embedding_size=128, input_size=224, width_mult=1., model_type=32):
        super(MobileNetV2, self).__init__()
        block_dwc = InvertedResidual_dwc
        input_channel = 64
        last_channel = 256
        interverted_residual_setting = [
            # t, c, n, s
            [1, 32, 1, 1],  # depthwise conv for first row
            [2, 32, 2, 1],
            [4, 32, 2, 1],
            [2, 32, 2, 2],
            [4, 32, 5, 1],
            [2, 32, 2, 2],
            [2, 32, 6, 2],
        ]
        if model_type != 32:
            for idx, irs in enumerate(interverted_residual_setting):
                irs[1] = model_type
                if idx == 2:
                    irs[3] = 2
                if idx == 3:
                    irs[3] = 1

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]

        # building inverted residual
        cnt = 0
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if cnt>1:
                    if i == n - 1:  # reduce the featuremap in the last.
                        self.features.append(block_dwc(input_channel, output_channel, s, expand_ratio=t, model_type=model_type))
                    else:
                        self.features.append(block_dwc(input_channel, output_channel, 1, expand_ratio=t, model_type=model_type))
                    input_channel = output_channel
                else:
                    if i == n - 1:  # reduce the featuremap in the last.
                        self.features.append(block_dwc(input_channel, output_channel, s, expand_ratio=t, model_type=model_type))
                    else:
                        self.features.append(block_dwc(input_channel, output_channel, 1, expand_ratio=t, model_type=model_type))
                    input_channel = output_channel

            cnt+=1

        # building last several layers
        self.features.append(gated_conv1x1(input_channel, self.last_channel))

        # make it nn.Sequential
        self.features_sequential = nn.Sequential(*self.features)

    def forward(self, x):
        x = self.features_sequential(x).view(-1, 256*4)

        return x