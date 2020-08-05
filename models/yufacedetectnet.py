import flops_counter
import flops_counter.nn as nn

class YuFaceDetectNet(nn.Module):
    def __init__(self):
        super(YuFaceDetectNet, self).__init__()
        self.num_classes = 2

        self.model1 = Conv_2layers(3, 32, 16, 2)
        self.model2 = Conv_2layers(16, 32, 32, 1)
        self.model3 = Conv_3layers(32, 64, 32, 64, 1)
        self.model4 = Conv_3layers(64, 128, 64, 128, 1)
        self.model5 = Conv_3layers(128, 256, 128, 256, 1)
        self.model6 = Conv_3layers(256, 256, 256, 256, 1)

        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)
        self.pool5 = nn.MaxPool2d(2)


        self.loc, self.conf = self.multibox(self.num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def multibox(self, num_classes):
        loc_layers = []
        conf_layers = []
        loc_layers += [nn.Conv2d(self.model3.out_channels, 3 * 14, kernel_size=3, padding=1, bias=True)]
        conf_layers += [nn.Conv2d(self.model3.out_channels, 3 * num_classes, kernel_size=3, padding=1, bias=True)]
        loc_layers += [nn.Conv2d(self.model4.out_channels, 2 * 14, kernel_size=3, padding=1, bias=True)]
        conf_layers += [nn.Conv2d(self.model4.out_channels, 2 * num_classes, kernel_size=3, padding=1, bias=True)]
        loc_layers += [nn.Conv2d(self.model5.out_channels, 2 * 14, kernel_size=3, padding=1, bias=True)]
        conf_layers += [nn.Conv2d(self.model5.out_channels, 2 * num_classes, kernel_size=3, padding=1, bias=True)]
        loc_layers += [nn.Conv2d(self.model6.out_channels, 3 * 14, kernel_size=3, padding=1, bias=True)]
        conf_layers += [nn.Conv2d(self.model6.out_channels, 3 * num_classes, kernel_size=3, padding=1, bias=True)]
        return nn.Sequential(*loc_layers), nn.Sequential(*conf_layers)

    def forward(self, x):

        detection_sources = list()
        loc_data = list()
        conf_data = list()

        x = self.model1(x)
        # x = F.max_pool2d(x, 2)
        x = self.pool1(x)
        x = self.model2(x)
        # x = F.max_pool2d(x, 2)
        x = self.pool2(x)
        x = self.model3(x)
        detection_sources.append(x)

        # x = F.max_pool2d(x, 2)
        x = self.pool3(x)
        x = self.model4(x)
        detection_sources.append(x)

        # x = F.max_pool2d(x, 2)
        x = self.pool4(x)
        x = self.model5(x)
        detection_sources.append(x)

        # x = F.max_pool2d(x, 2)
        x = self.pool5(x)
        x = self.model6(x)
        detection_sources.append(x)

        for (x, l, c) in zip(detection_sources, self.loc, self.conf):
            # loc_data.append(l(x).permute(0, 2, 3, 1).contiguous())
            # conf_data.append(c(x).permute(0, 2, 3, 1).contiguous())
            loc_data.append(l(x).permute(0, 2, 3, 1))
            conf_data.append(c(x).permute(0, 2, 3, 1))

        loc_data = flops_counter.cat([o.view(o.size(0), -1) for o in loc_data], 1)
        conf_data = flops_counter.cat([o.view(o.size(0), -1) for o in conf_data], 1)

        output = (loc_data.view(loc_data.size(0), -1, 14),
                self.softmax(conf_data.view(conf_data.size(0), -1, self.num_classes)))

        return output

    @property
    def name(self):
        return self._get_name()


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        # x = F.relu(x, inplace=True)
        x = self.relu(x)
        return x

class Conv_2layers(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride, **kwargs):
        super(Conv_2layers, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = ConvBNReLU(in_channels, mid_channels, 3, stride, 1, **kwargs)
        self.conv2 = ConvBNReLU(mid_channels, out_channels, 1, 1, 0, **kwargs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Conv_3layers(nn.Module):
    def __init__(self, in_channels, mid1_channels, mid2_channels, out_channels, stride, **kwargs):
        super(Conv_3layers, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = ConvBNReLU(in_channels, mid1_channels, 3, stride, 1, **kwargs)
        self.conv2 = ConvBNReLU(mid1_channels, mid2_channels, 1, 1, 0, **kwargs)
        self.conv3 = ConvBNReLU(mid2_channels, out_channels, 3, 1, 1, **kwargs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x