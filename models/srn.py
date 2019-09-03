import flops_counter
import flops_counter.nn as nn
import flops_counter.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.eltadd = nn.EltAdd()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.eltadd(out, residual)
        out = self.relu(out)

        return out

class RFE(nn.Module):

    def __init__(self, in_planes=256, out_planes=256):
        super(RFE, self).__init__()
        self.out_channels = out_planes
        self.inter_channels = int(in_planes / 4)

        self.branch0 = nn.Sequential(
                nn.Conv2d(in_planes, self.inter_channels, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.inter_channels, self.inter_channels, kernel_size=(1, 5), stride=1, padding=(0, 2)),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.inter_channels, self.inter_channels, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True)
                )
        self.branch1 = nn.Sequential(
                nn.Conv2d(in_planes, self.inter_channels, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.inter_channels, self.inter_channels, kernel_size=(5, 1), stride=1, padding=(2, 0)),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.inter_channels, self.inter_channels, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True)
                )
        self.branch2 = nn.Sequential(
                nn.Conv2d(in_planes, self.inter_channels, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.inter_channels, self.inter_channels, kernel_size=(1, 3), stride=1, padding=(0, 1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.inter_channels, self.inter_channels, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True)
                )
        self.branch3 = nn.Sequential(
                nn.Conv2d(in_planes, self.inter_channels, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.inter_channels, self.inter_channels, kernel_size=(3, 1), stride=1, padding=(1, 0)),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.inter_channels, self.inter_channels, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True)
                )
        self.cated_conv = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True)
                )

        self.eltadd = nn.EltAdd()

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = flops_counter.cat((x0, x1, x2, x3), 1)
        out = self.cated_conv(out)
        out = self.eltadd(out, x)

        return out

class SRN(nn.Module):
    def __init__(self):
        super(SRN, self).__init__()

        block = Bottleneck
        layers =  [3, 4, 6, 3]
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.layer5 = nn.Conv2d(2048, 1024, kernel_size=3, stride=2, padding=1)
        self.layer6 = nn.Conv2d(1024, 256, kernel_size=3, stride=2, padding=1)

        self.c5_lateral = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
        self.c4_lateral = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.c3_lateral = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.c2_lateral = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        self.eltadd = nn.EltAdd()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.p7_conv = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.p6_conv = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.p5_conv = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.p4_conv = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.p3_conv = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.p2_conv = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.c7_conv = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.c6_conv = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.c5_conv = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
        self.c4_conv = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.c3_conv = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.c2_conv = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        # subnet_first stage
        num_anchors = 2 * 1
        self.cls_subnet = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            RFE(256, 256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.box_subnet = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            RFE(256, 256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.cls_subnet_pred = nn.Conv2d(256, num_anchors * 1, kernel_size=1, stride=1, padding=0)
        self.box_subnet_pred = nn.Conv2d(256, num_anchors * 4, kernel_size=1, stride=1, padding=0)

        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    @property
    def name(self):
        return self._get_name() + '_ResNet50'

    def feature_extractor(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        c6 = self.layer5(c5)
        c7 = self.layer6(c6)

        c2_r = self.c2_conv(c2)
        c3_r = self.c3_conv(c3)
        c4_r = self.c4_conv(c4)
        c5_r = self.c5_conv(c5)
        c6_r = self.c6_conv(c6)
        c7_r = self.c7_conv(c7)

        c5_lateral = self.c5_lateral(c5)
        p5 = self.p5_conv(c5_lateral)
        p6 = self.p6_conv(c5_lateral)
        p7 = self.p7_conv(p6)

        c4_lateral = self.c4_lateral(c4)
        c5_lateral_up = self.upsample(c5_lateral)
        sum_4 = self.eltadd(c4_lateral, c5_lateral_up)
        p4 = self.p4_conv(sum_4)

        c3_lateral = self.c3_lateral(c3)
        sum_4_up = self.upsample(sum_4)
        sum_3 = self.eltadd(sum_4_up, c3_lateral)
        p3 = self.p3_conv(sum_3)

        c2_lateral = self.c2_lateral(c2)
        sum_3_up = self.upsample(sum_3)
        sum_2 = self.eltadd(sum_3_up, c2_lateral)
        p2 = self.p2_conv(sum_2)

        return (c2_r, c3_r, c4_r, c5_r, c6_r, c7_r), (p2, p3, p4, p5, p6, p7)

    def rpn(self, x):
        cls_feature = self.cls_subnet(x)
        box_feature = self.box_subnet(x)
        rpn_pred_cls = self.cls_subnet_pred(cls_feature)
        rpn_pred_loc = self.box_subnet_pred(box_feature)
        return rpn_pred_cls, rpn_pred_loc

    def forward(self, x):
        rpn_pred_fs = []
        rpn_pred_ss = []

        pyramid_features_fs, pyramid_features_ss = self.feature_extractor(x)

        for feature in pyramid_features_fs:
            rpn_pred_fs.append(self.rpn(feature))
        for feature in pyramid_features_ss:
            rpn_pred_ss.append(self.rpn(feature))

        fpn_level = len(pyramid_features_fs)
        for level in range(fpn_level):
            rpn_pred_cls_fs = rpn_pred_fs[level][0]
            rpn_pred_cls_ss = rpn_pred_ss[level][0]
            rpn_pred_cls_fs = self.sigmoid(rpn_pred_cls_fs)
            rpn_pred_cls_ss = self.sigmoid(rpn_pred_cls_ss)

        return rpn_pred_cls_fs, rpn_pred_cls_ss