import numpy as np

import flops_counter
import flops_counter.nn.functional as F
import flops_counter.nn as nn

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Inception(nn.Module):
  def __init__(self):
    super(Inception, self).__init__()
    self.branch1x1 = BasicConv2d(128, 32, kernel_size=1, padding=0)
    self.branch1x1_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
    self.branch1x1_2 = BasicConv2d(128, 32, kernel_size=1, padding=0)
    self.branch3x3_reduce = BasicConv2d(128, 24, kernel_size=1, padding=0)
    self.branch3x3 = BasicConv2d(24, 32, kernel_size=3, padding=1)
    self.branch3x3_reduce_2 = BasicConv2d(128, 24, kernel_size=1, padding=0)
    self.branch3x3_2 = BasicConv2d(24, 32, kernel_size=3, padding=1)
    self.branch3x3_3 = BasicConv2d(32, 32, kernel_size=3, padding=1)

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

class CRelu(nn.Module):
  def __init__(self, in_channels, out_channels, **kwargs):
    super(CRelu, self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
    self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    x = flops_counter.cat([x, -x], 1)
    # x = F.relu(x, inplace=True)
    x = self.relu(x)
    return x

class FaceBoxes(nn.Module):
    def __init__(self):
        super(FaceBoxes, self).__init__()
        self.num_classes = 2

        self.conv1 = CRelu(3, 24, kernel_size=7, stride=4, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = CRelu(48, 64, kernel_size=5, stride=2, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception1 = Inception()
        self.inception2 = Inception()
        self.inception3 = Inception()

        self.conv3_1 = BasicConv2d(128, 128, kernel_size=1, stride=1, padding=0)
        self.conv3_2 = BasicConv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.conv4_1 = BasicConv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.conv4_2 = BasicConv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.loc, self.conf = self.multibox(self.num_classes)

        self.softmax = nn.Softmax(dim=-1)

    def multibox(self, num_classes):
        loc_layers = []
        conf_layers = []
        loc_layers += [nn.Conv2d(128, 21 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(128, 21 * num_classes, kernel_size=3, padding=1)]
        loc_layers += [nn.Conv2d(256, 1 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(256, 1 * num_classes, kernel_size=3, padding=1)]
        loc_layers += [nn.Conv2d(256, 1 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(256, 1 * num_classes, kernel_size=3, padding=1)]
        return nn.Sequential(*loc_layers), nn.Sequential(*conf_layers)

    def forward(self, x):
        detection_sources = list()
        loc = list()
        conf = list()

        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        detection_sources.append(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        detection_sources.append(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        detection_sources.append(x)

        for (x, l, c) in zip(detection_sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1))
            conf.append(c(x).permute(0, 2, 3, 1))

        loc = flops_counter.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = flops_counter.cat([o.view(o.size(0), -1) for o in conf], 1)

        output = (loc.view(loc.size(0), -1, 4),
                    self.softmax(conf.view(conf.size(0), -1, self.num_classes)))

        return output

    @property
    def name(self):
        return self._get_name()