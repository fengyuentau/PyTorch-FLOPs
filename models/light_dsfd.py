import flops_counter
import flops_counter.nn as nn
import flops_counter.nn.functional as F

class light_DSFD(nn.Module):
    def __init__(self):
        super(light_DSFD, self).__init__()

        self.conv1 = CRelu(3, 32, kernel_size=7, stride=4, padding=3)
        self.conv3 = CRelu(64, 64, kernel_size=5, stride=2, padding=2)

        self.inception1 = Inception2d(64)
        self.inception2 = Inception2d(64)
        self.inception3 = Inception2d(128)
        self.inception4 = Inception2d(128)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv5_1 = BasicConv2d(128, 128, kernel_size=1, stride=1, padding=0)
        self.conv5_2 = BasicConv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv6_1 = BasicConv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.conv6_2 = BasicConv2d(128, 256, kernel_size=3, stride=2, padding=1)

        fpn_in = [64, 64, 128, 128, 256, 256]
        cpm_in = [64, 64, 64, 64, 64, 64]
        fpn_channel = 64
        cpm_channels = 64
        output_channels = cpm_in

        # fpn
        self.smooth3 = nn.Conv2d( fpn_channel, fpn_channel, kernel_size=1, stride=1, padding=0)
        self.smooth2 = nn.Conv2d( fpn_channel, fpn_channel, kernel_size=1, stride=1, padding=0)
        self.smooth1 = nn.Conv2d( fpn_channel, fpn_channel, kernel_size=1, stride=1, padding=0)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.eltmul = nn.EltMul()

        self.latlayer6 = nn.Conv2d( fpn_in[5], fpn_channel, kernel_size=1, stride=1, padding=0)
        self.latlayer5 = nn.Conv2d( fpn_in[4], fpn_channel, kernel_size=1, stride=1, padding=0)
        self.latlayer4 = nn.Conv2d( fpn_in[3], fpn_channel, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( fpn_in[2], fpn_channel, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( fpn_in[1], fpn_channel, kernel_size=1, stride=1, padding=0)
        self.latlayer1 = nn.Conv2d( fpn_in[0], fpn_channel, kernel_size=1, stride=1, padding=0)

        # cpm
        self.cpm1 = Inception2d(cpm_in[0])
        self.cpm2 = Inception2d(cpm_in[1])
        self.cpm3 = Inception2d(cpm_in[2])
        self.cpm4 = Inception2d(cpm_in[3])
        self.cpm5 = Inception2d(cpm_in[4])
        self.cpm6 = Inception2d(cpm_in[5])

        face_head = face_multibox(output_channels, [1, 1, 1, 1, 1, 1], 2 , cpm_channels)  
        self.loc = nn.ModuleList(face_head[0])
        self.conf = nn.ModuleList(face_head[1])

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        loc = list()
        conf = list()

        conv1_x = self.inception1(self.conv1(x))
        conv2_x = self.inception2(self.maxpool1(conv1_x))
        conv3_x = self.inception3(self.conv3(conv2_x))
        conv4_x = self.inception4(self.maxpool2(conv3_x))
        conv5_x = self.conv5_2(self.conv5_1(conv4_x))
        conv6_x = self.conv6_2(self.conv6_1(conv5_x))

        # fpn
        conv6_x = self.latlayer6(conv6_x)
        conv5_x = self.latlayer5(conv5_x)
        conv4_x = self.latlayer4(conv4_x)
        conv3_x = self.latlayer3(conv3_x)
        conv2_x = self.latlayer2(conv2_x)
        conv1_x = self.latlayer1(conv1_x)

        conv4_x_up = self.upsample(conv4_x)
        if conv4_x_up.value[2] != conv3_x.value[2] or conv4_x_up.value[3] != conv3_x.value[3]:
            pad = (0, conv3_x.value[3] - conv4_x_up.value[3], 0, conv3_x.value[2] - conv4_x_up.value[2])
            conv4_x_up = F.pad(conv4_x_up, pad)
        conv3_x = self.smooth3(self.eltmul(conv4_x_up , conv3_x))

        conv3_x_up = self.upsample(conv2_x)
        if conv3_x_up.value[2] != conv2_x.value[2] or conv3_x_up.value[3] != conv2_x.value[3]:
            pad = (0, conv2_x.value[3] - conv3_x_up.value[3], 0, conv2_x.value[2] - conv3_x_up.value[2])
            conv3_x_up = F.pad(conv3_x_up, pad)
        conv2_x = self.smooth2(self.eltmul(conv3_x_up , conv2_x))

        conv2_x_up = self.upsample(conv2_x)
        if conv2_x_up.value[2] != conv1_x.value[2] or conv2_x_up.value[3] != conv1_x.value[3]:
            pad = (0, conv1_x.value[3] - conv2_x_up.value[3], 0, conv1_x.value[2] - conv2_x_up.value[2])
            conv2_x_up = F.pad(conv2_x_up, pad)
        conv1_x = self.smooth1(self.eltmul(conv2_x_up , conv1_x))

        sources = [conv1_x, conv2_x, conv3_x, conv4_x, conv5_x, conv6_x]
        # cpm
        sources[0] = self.cpm1(sources[0])
        sources[1] = self.cpm2(sources[1])
        sources[2] = self.cpm3(sources[2])
        sources[3] = self.cpm4(sources[3])
        sources[4] = self.cpm5(sources[4])
        sources[5] = self.cpm6(sources[5])

        # head
        featuremap_size = []
        for (x, l, c) in zip(sources, self.loc, self.conf):
            featuremap_size.append([x.size(2), x.size(3)])
            loc.append(l(x).permute(0, 2, 3, 1))
            conf.append(c(x).permute(0, 2, 3, 1))

        face_loc = flops_counter.cat([o.view(o.size(0), -1) for o in loc], 1)
        face_conf = flops_counter.cat([o.view(o.size(0), -1) for o in conf], 1)

        return face_loc.view(face_loc.size(0), -1, 4), self.softmax(face_conf.view(face_conf.size(0), -1, 2))

    @property
    def name(self):
        return self._get_name()


class Inception2d(nn.Module):
    def __init__(self, in_channels , out_channels=None):
        super(Inception2d, self).__init__()

        mid_channels = int(in_channels/8)
        out_channels = int(in_channels/4)
        self.branch1x1 = BasicConv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.branch1x1_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch1x1_2 = BasicConv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.branch3x3_reduce = BasicConv2d(in_channels, mid_channels, kernel_size=1, padding=0)
        self.branch3x3 = BasicConv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.branch3x3_reduce_2 = BasicConv2d(in_channels, mid_channels, kernel_size=1, padding=0)
        self.branch3x3_2 = BasicConv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.branch3x3_3 = BasicConv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        # branch1x1_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch1x1_pool = self.branch1x1_pool(x)
        branch1x1_2 = self.branch1x1_2(branch1x1_pool)

        branch3x3_reduce = self.branch3x3_reduce(x)
        branch3x3 = self.branch3x3(branch3x3_reduce)

        branch3x3_reduce_2 = self.branch3x3_reduce_2(x)
        branch3x3_2 = self.branch3x3_2(branch3x3_reduce_2)
        branch3x3_3 = self.branch3x3_3(branch3x3_2)

        outputs = [branch1x1, branch1x1_2, branch3x3, branch3x3_3]
        return flops_counter.cat(outputs, 1)

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class CRelu(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = flops_counter.cat([x, -x], 1)
        # x = F.relu(x, inplace=True)
        x = self.relu(x)
        return x

def face_multibox(output_channels, mbox_cfg, num_classes , cpm_c):
    loc_layers = []
    conf_layers = []
    for k, v in enumerate(output_channels):
        input_channels = cpm_c
        loc_output = 4
        conf_output = 2
        loc_layers  += [ DeepHeadModule(input_channels, mbox_cfg[k] * loc_output) ]
        conf_layers += [ DeepHeadModule(input_channels, mbox_cfg[k] * conf_output)]
    return (loc_layers, conf_layers)

class DeepHeadModule(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(DeepHeadModule , self).__init__()
        self._input_channels = input_channels
        self._mid_channels = 16
        self._output_channels = output_channels
        self.conv1 = BasicConv2d(self._input_channels, self._mid_channels, kernel_size=1, dilation=1, stride=1, padding=0)
        self.conv2 = BasicConv2d(self._mid_channels, self._mid_channels, kernel_size=3, dilation=1, stride=1, padding=1)
        self.conv3 = nn.Conv2d(self._mid_channels, self._output_channels, kernel_size=1, dilation=1, stride=1, padding=0)
    def forward(self, x):
        #return self.conv3( F.relu(self.conv2( F.relu(self.conv1(x), inplace=True) )inplace=True) )
        return self.conv3(self.conv2(self.conv1(x)))