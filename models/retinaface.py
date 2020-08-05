import flops_counter
import flops_counter.nn as nn
import flops_counter.nn.functional as F

import vision.models._utils as _utils
from vision.models import resnet50, MobileNetV1
from vision.models.mobilenet import conv_bn, conv_dw

class RetinaFace(nn.Module):
    def __init__(self, backbone='ResNet50'):
        super(RetinaFace, self).__init__()

        if backbone.lower() == 'resnet50':
            self.backbone = resnet50()
            self.return_layers = { 'layer2': 1, 'layer3': 2, 'layer4': 3 }
            self.in_channels_stage2 = 256
            self.out_channels = 256
        elif backbone.lower() == 'mobilenetv1':
            self.backbone = MobileNetV1()
            self.return_layers = { 'stage1': 1, 'stage2': 2, 'stage3': 3 }
            self.in_channels_stage2 = 32
            self.out_channels = 64
        else:
            raise NotImplementedError

        self.body = _utils.IntermediateLayerGetter(self.backbone, self.return_layers)
        self.in_channels_list = [
            self.in_channels_stage2 * 2,
            self.in_channels_stage2 * 4,
            self.in_channels_stage2 * 8
        ]
        self.fpn = FPN(self.in_channels_list, self.out_channels)
        self.ssh1 = SSH(self.out_channels, self.out_channels)
        self.ssh2 = SSH(self.out_channels, self.out_channels)
        self.ssh3 = SSH(self.out_channels, self.out_channels)

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=self.out_channels)
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=self.out_channels)
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=self.out_channels)
        self.softmax = nn.Softmax(dim=-1)

    def _make_class_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num))
        return classhead

    def _make_bbox_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num))
        return bboxhead

    def _make_landmark_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels,anchor_num))
        return landmarkhead

    def forward(self, x):
        out = self.body(x)

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = flops_counter.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = flops_counter.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)],dim=1)
        ldm_regressions = flops_counter.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        output = (bbox_regressions, self.softmax(classifications), ldm_regressions)
        return output

    @property
    def name(self):
        return self._get_name() + '-' + self.backbone._get_name()

class FPN(nn.Module):
    def __init__(self,in_channels_list,out_channels):
        super(FPN,self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride = 1, leaky = leaky)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride = 1, leaky = leaky)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride = 1, leaky = leaky)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.eltadd = nn.EltAdd()

        self.merge1 = conv_bn(out_channels, out_channels, leaky = leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky = leaky)


    def forward(self, input):
        # names = list(input.keys())
        input = list(input.values())

        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2])

        # up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest") # upsample by 2
        up3 = self.upsample(output3)
        if up3.value[2] != output2.value[2] or up3.value[3] != output2.value[3]:
            pad = (0, output2.value[3] - up3.value[3], 0, output2.value[2] - up3.value[2])
            up3 = F.pad(up3, pad)
        output2 = self.eltadd(output2, up3)
        output2 = self.merge2(output2)

        # up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest") # upsample by 2
        up2 = self.upsample(output3)
        if up2.value[2] != output1.value[2] or up2.value[3] != output1.value[3]:
            pad = (0, output1.value[3] - up2.value[3], 0, output1.value[2] - up2.value[2])
            up2 = F.pad(up2, pad)
        output1 = self.eltadd(output1, up2)
        output1 = self.merge1(output1)

        out = [output1, output2, output3]
        return out

def conv_bn1X1(inp, oup, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if (out_channel <= 64):
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel//2, stride=1)

        self.conv5X5_1 = conv_bn(in_channel, out_channel//4, stride=1, leaky = leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

        self.conv7X7_2 = conv_bn(out_channel//4, out_channel//4, stride=1, leaky = leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)
        self.relu = nn.ReLU()

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = flops_counter.cat([conv3X3, conv5X5, conv7X7], dim=1)
        # out = F.relu(out)
        out = self.relu(out)
        return out

def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )

class ClassHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(ClassHead,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        # out = out.permute(0,2,3,1).contiguous()
        out = out.permute(0, 2, 3, 1)
        
        return out.view(out.value[0], -1, 2)

class BboxHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        # out = out.permute(0,2,3,1).contiguous()
        out = out.permute(0, 2, 3, 1)

        return out.view(out.value[0], -1, 4)

class LandmarkHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(LandmarkHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*10,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        #out = out.permute(0,2,3,1).contiguous()
        out = out.permute(0, 2, 3, 1)

        return out.view(out.value[0], -1, 10)