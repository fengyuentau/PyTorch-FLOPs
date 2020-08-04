import flops_counter
import flops_counter.nn as nn

class ULFG(nn.Module):
    def __init__(self, mode='slim'):
        super(ULFG, self).__init__()
        self.mode = mode

        self.base_channel = 8 * 2
        self.backbone = nn.Sequential(
            _conv_bn(3, self.base_channel, 2),  # 160*120
            _conv_dw(self.base_channel, self.base_channel * 2, 1),
            _conv_dw(self.base_channel * 2, self.base_channel * 2, 2),  # 80*60
            _conv_dw(self.base_channel * 2, self.base_channel * 2, 1),
            _conv_dw(self.base_channel * 2, self.base_channel * 4, 2),  # 40*30
            _conv_dw(self.base_channel * 4, self.base_channel * 4, 1),
            _conv_dw(self.base_channel * 4, self.base_channel * 4, 1),
            _conv_dw(self.base_channel * 4, self.base_channel * 4, 1),
            _conv_dw(self.base_channel * 4, self.base_channel * 8, 2),  # 20*15
            _conv_dw(self.base_channel * 8, self.base_channel * 8, 1),
            _conv_dw(self.base_channel * 8, self.base_channel * 8, 1),
            _conv_dw(self.base_channel * 8, self.base_channel * 16, 2),  # 10*8
            _conv_dw(self.base_channel * 16, self.base_channel * 16, 1)
        )
        if self.mode == 'rfb':
            self.backbone[7] = BasicRFB(self.base_channel * 4, self.base_channel * 4, stride=1, scale=1.0)

        self.source_layer_indexes = [8, 11, 13]
        self.extras = nn.Sequential(
            nn.Conv2d(in_channels=self.base_channel * 16, out_channels=self.base_channel * 4, kernel_size=1),
            nn.ReLU(),
            _seperable_conv2d(in_channels=self.base_channel * 4, out_channels=self.base_channel * 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.regression_headers = nn.ModuleList([
            _seperable_conv2d(in_channels=self.base_channel * 4, out_channels=3 * 4, kernel_size=3, padding=1),
            _seperable_conv2d(in_channels=self.base_channel * 8, out_channels=2 * 4, kernel_size=3, padding=1),
            _seperable_conv2d(in_channels=self.base_channel * 16, out_channels=2 * 4, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=self.base_channel * 16, out_channels=3 * 4, kernel_size=3, padding=1)
        ])
        self.classification_headers = nn.ModuleList([
            _seperable_conv2d(in_channels=self.base_channel * 4, out_channels=3 * 2, kernel_size=3, padding=1),
            _seperable_conv2d(in_channels=self.base_channel * 8, out_channels=2 * 2, kernel_size=3, padding=1),
            _seperable_conv2d(in_channels=self.base_channel * 16, out_channels=2 * 2, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=self.base_channel * 16, out_channels=3 * 2, kernel_size=3, padding=1)
        ])
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        confidences = []
        locations = []
        start_layer_index = 0
        header_index = 0
        end_layer_index = 0
        for end_layer_index in self.source_layer_indexes:
            for layer in self.backbone[start_layer_index: end_layer_index]:
                x = layer(x)
            y = x
            start_layer_index = end_layer_index
            confidence, location = self.compute_header(header_index, y)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)

        for layer in self.backbone[end_layer_index:]:
            x = layer(x)

        layer = self.extras
        x = layer(x)
        confidence, location = self.compute_header(header_index, x)
        header_index += 1
        confidences.append(confidence)
        locations.append(location)

        confidences = flops_counter.cat(confidences, 1)
        locations = flops_counter.cat(locations, 1)
        confidences = self.softmax(confidences)
        return locations, confidences

    def compute_header(self, i, x):
        confidence = self.classification_headers[i](x)
        confidence = confidence.permute(0, 2, 3, 1)
        confidence = confidence.view(confidence.size(0), -1, 2)

        location = self.regression_headers[i](x)
        location = location.permute(0, 2, 3, 1)
        location = location.view(location.size(0), -1, 4)

        return confidence, location

    @property
    def name(self):
        print(self.backbone)
        return self._get_name() + '-' + self.mode


def _conv_bn(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True)
    )


def _conv_dw(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def _seperable_conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, groups=in_channels, stride=stride, padding=padding),
        nn.ReLU(),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
    )


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        if bn:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False)
            self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            self.relu = nn.ReLU(inplace=True) if relu else None
        else:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)
            self.bn = None
            self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class BasicRFB(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8, vision=1, groups=1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce

        self.branch0 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 1, dilation=vision + 1, relu=False, groups=groups)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 2, dilation=vision + 2, relu=False, groups=groups)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1, groups=groups),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1, groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 4, dilation=vision + 4, relu=False, groups=groups)
        )

        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)
        self.eltadd = nn.EltAdd()

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = flops_counter.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        # out = out * self.scale + short
        out = self.eltadd(out * self.scale, short)
        out = self.relu(out)

        return out