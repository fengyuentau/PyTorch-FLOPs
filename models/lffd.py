import numpy as np

import flops_counter
import flops_counter.nn.functional as F
import flops_counter.nn as nn

class ResBlock(nn.Module):
    def __init__(self, channels, det_out=False):
        super(ResBlock, self).__init__()

        self.channels = channels
        self.det_out = det_out

        self.relu = nn.ReLU()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, stride=1, padding=1)
        )

        self.eltadd = nn.EltAdd()

    def forward(self, x):
        x_relu = self.relu(x)
        out = self.block(x_relu)
        out = self.eltadd(out, x)
        if self.det_out:
            return out, x
        return out

class DetBlock(nn.Module):
    def __init__(self, in_channels):
        super(DetBlock, self).__init__()
        self.in_channels = in_channels
        self.det_channels = 128

        self.det_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.det_channels, kernel_size=1, stride=1, padding=0)
        self.det_relu = nn.ReLU()

        self.bbox_conv = nn.Conv2d(in_channels=self.det_channels, out_channels=self.det_channels, kernel_size=1, stride=1, padding=0)
        self.bbox_relu = nn.ReLU()
        self.bbox_out_conv = nn.Conv2d(in_channels=self.det_channels, out_channels=4, kernel_size=1, stride=1, padding=0)

        self.score_conv = nn.Conv2d(in_channels=self.det_channels, out_channels=self.det_channels, kernel_size=1, stride=1, padding=0)
        self.score_relu = nn.ReLU()
        self.score_out_conv = nn.Conv2d(in_channels=self.det_channels, out_channels=2, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.det_relu(self.det_conv(x))
        # bbox
        bbox_f = self.bbox_relu(self.bbox_conv(x))
        out_bbox = self.bbox_out_conv(bbox_f)
        # score
        score_f = self.score_relu(self.score_conv(x))
        out_score = self.softmax(self.score_out_conv(score_f))

        return out_bbox, out_score

class LFFDv1(nn.Module):
    def __init__(self):
        super(LFFDv1, self).__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=0),      # downsample by 2
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=0),     # downsample by 2
            ResBlock(64),
            ResBlock(64),
            ResBlock(64)
        )

        self.rb1 = ResBlock(64, det_out=True)
        self.det1 = DetBlock(64)

        self.relu_conv10 = nn.ReLU()
        self.conv11 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=0)
        self.rb2 = ResBlock(64)
        self.det2 = DetBlock(64)

        self.rb3 = ResBlock(64, det_out=True)
        self.det3 = DetBlock(64)

        self.relu_conv15 = nn.ReLU()
        self.conv16 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=0)
        self.rb4 = ResBlock(128)
        self.det4 = DetBlock(64)

        self.relu_conv18 = nn.ReLU()
        self.conv19 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=0)
        self.rb5 = ResBlock(128)
        self.det5 = DetBlock(128)

        self.rb6 = ResBlock(128, det_out=True)
        self.det6 = DetBlock(128)

        self.rb7 = ResBlock(128, det_out=True)
        self.det7 = DetBlock(128)

        self.relu_conv25 = nn.ReLU()
        self.det8 = DetBlock(128)

    def forward(self, x):
        x = self.backbone(x)

        x, relu_conv8 = self.rb1(x)
        det1 = self.det1(relu_conv8)

        relu_conv10 = self.relu_conv10(x)
        det2 = self.det2(relu_conv10)
        x = self.rb2(self.conv11(relu_conv10))

        x, relu_conv13 = self.rb3(x)
        det3 = self.det3(relu_conv13)

        relu_conv15 = self.relu_conv15(x)
        det4 = self.det4(relu_conv15)
        x = self.rb4(self.conv16(relu_conv15))

        relu_conv18 = self.relu_conv18(x)
        det5 = self.det5(relu_conv18)
        x = self.rb5(self.conv19(relu_conv18))

        x, relu_conv21 = self.rb6(x)
        det6 = self.det6(relu_conv21)

        x, relu_conv23 = self.rb7(x)
        det7 = self.det7(relu_conv23)

        relu_conv25 = self.relu_conv25(x)
        det8 = self.det8(relu_conv25)

        return det1, det2, det3, det4, det5, det6, det7, det8

    @property
    def name(self):
        return self._get_name() + 'v1'


class LFFDv2(nn.Module):
    def __init__(self):
        super(LFFDv2, self).__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=0),              # downsample by 2
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=0),             # downsample by 2
            ResBlock(64),
            ResBlock(64),
            ResBlock(64)
        )

        self.relu_conv8 = nn.ReLU()
        self.conv9 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=0)     # downsample by 2
        self.rb1 = ResBlock(64)
        self.det1 = DetBlock(64)

        self.relu_conv11 = nn.ReLU()
        self.conv12 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=0)    # downsample by 2
        self.rb2 = ResBlock(64)
        self.det2 = DetBlock(64)

        self.relu_conv14 = nn.ReLU()
        self.conv15 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=0)    # downsample by 2
        self.rb3 = ResBlock(128)
        self.det3 = DetBlock(64)

        self.relu_conv17 = nn.ReLU()
        self.conv18 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=0)    # downsample by 2
        self.rb4 = ResBlock(128)
        self.det4 = DetBlock(128)

        self.relu_conv20 = nn.ReLU()
        self.det5 = DetBlock(128)

    def forward(self, x):
        x = self.backbone(x)

        relu_conv8 = self.relu_conv8(x)
        det1 = self.det1(relu_conv8)
        x = self.rb1(self.conv9(relu_conv8))

        relu_conv11 = self.relu_conv11(x)
        det2 = self.det2(relu_conv11)
        x = self.rb2(self.conv12(relu_conv11))

        relu_conv14 = self.relu_conv14(x)
        det3 = self.det3(relu_conv14)
        x = self.rb3(self.conv15(relu_conv14))

        relu_conv17 = self.relu_conv17(x)
        det4 = self.det4(relu_conv17)
        x = self.rb4(self.conv18(relu_conv17))

        relu_conv20 = self.relu_conv20(x)
        det5 = self.det5(relu_conv20)

        return det1, det2, det3, det4, det5


    @property
    def name(self):
        return self._get_name() + 'v2'