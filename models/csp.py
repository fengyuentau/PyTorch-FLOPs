import sys
sys.path.append('..')

import flops_counter
import flops_counter.nn as nn

class IdentityBlock(object):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(IdentityBlock, self).__init__()

        out_channels_1, out_channels_2, out_channels_3 = out_channels//4, out_channels//4, out_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels_1, kernel_size=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels_1)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels_1, out_channels_2, kernel_size=(kernel_size, kernel_size), padding=(padding, padding), dilation=(dilation, dilation), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels_2)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels_2, out_channels_3, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels_3)

        self.eltadd = nn.EltAdd()
        self.relu_f = nn.ReLU(inplace=True)

    def __call__(self, x):
        flops = int(0)
        identity = x

        out, flops_t = self.conv1(out)
        flops += flops_t
        out, flops = self.bn1(out)
        flops += flops_t
        out, flops = self.relu1(out)
        flops += flops_t

        out, flops = self.conv2(out)
        flops += flops_t
        out, flops = self.bn2(out)
        flops += flops_t
        out, flops = self.relu2(out)
        flops += flops_t

        out, flops = self.conv3(out)
        flops += flops_t
        out, flops = self.bn3(out)
        flops += flops_t

        out, flops = self.eltadd(identity, out)
        flops += flops_t
        out, flops = self.relu_f(out)
        flops += flops_t

        return out, flops

class ConvBlock(object):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(IdentityBlock, self).__init__()

        out_channels_1, out_channels_2, out_channels_3 = out_channels//4, out_channels//4, out_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels_1, kernel_size=(1, 1), stride=(stride, stride), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels_1)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels_1, out_channels_2, kernel_size=(kernel_size, kernel_size), padding=(padding, padding), dilation=(dilation, dilation), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels_2)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels_2, out_channels_3, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels_3)

        self.conv_shortcut = nn.Conv2d(in_channels, out_channels_3, kernel_size=(1, 1), stride=(stride, stride), bias=False)
        self.bn_shortcut = nn.BatchNorm2d(out_channels_3)

        self.eltadd = nn.EltAdd()
        self.relu_f = nn.ReLU(inplace=True)

    def __call__(self, x):
        flops = int(0)
        identity = x

        out, flops_t = self.conv1(out)
        flops += flops_t
        out, flops_t = self.bn1(out)
        flops += flops_t
        out, flops_t = self.relu1(out)
        flops += flops_t

        out, flops_t = self.conv2(out)
        flops += flops_t
        out, flops_t = self.bn2(out)
        flops += flops_t
        out, flops_t = self.relu2(out)
        flops += flops_t

        out, flops_t = self.conv3(out)
        flops += flops_t
        out, flops_t = self.bn3(out)
        flops += flops_t

        identity, flops_t = self.conv_shortcut(x)
        flops += flops_t
        identity, flops_t = self.bn_shortcut(identity)
        flops += flops_t

        out, flops_t = self.eltadd(identity, out)
        flops += flops_t
        out, flops_t = self.relu_f(out)
        flops += flops_t

        return out, flops

class CSP(object):
    def __init__(self):
        super(CSP, self).__init__()
        #####
        # Backbone
        #####
        # build resnet50
        # base
        self.base_conv = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.base_bn = nn.BatchNorm2d(64)
        self.base_relu = nn.ReLU(inplace=True)
        self.base_maxpooling = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        # layer1
        l1ic = [64, 256, 256]
        l1oc = [256, 256, 256]
        self.layer1_bottleneck0 = ConvBlock(    in_channels=l1ic[0], out_channels=l1oc[0], kernel_size=3, padding=1)
        self.layer1_bottleneck1 = IdentityBlock(in_channels=l1ic[1], out_channels=l1oc[1], kernel_size=3, padding=1)
        self.layer1_bottleneck2 = IdentityBlock(in_channels=l1ic[2], out_channels=l1oc[2], kernel_size=3, padding=1)
        # layer2
        l2ic = [256, 512, 512, 512]
        l2oc = [512, 512, 512, 512]
        self.layer2_bottleneck0 = ConvBlock(    in_channels=l2ic[0], out_channels=l2oc[0], kernel_size=3, stride=2, padding=1)
        self.layer2_bottleneck1 = IdentityBlock(in_channels=l2ic[1], out_channels=l2oc[1], kernel_size=3, padding=1)
        self.layer2_bottleneck2 = IdentityBlock(in_channels=l2ic[2], out_channels=l2oc[2], kernel_size=3, padding=1)
        self.layer2_bottleneck3 = IdentityBlock(in_channels=l2ic[3], out_channels=l2oc[3], kernel_size=3, padding=1)
        # layer3
        l3ic = [512, 1024, 1024, 1024, 1024, 1024]
        l3oc = [1024, 1024, 1024, 1024, 1024, 1024]
        self.layer3_bottleneck0 = ConvBlock(    in_channels=l3ic[0], out_channels=l3oc[0], kernel_size=3, stride=2, padding=1)
        self.layer3_bottleneck1 = IdentityBlock(in_channels=l3ic[1], out_channels=l3oc[1], kernel_size=3, padding=1)
        self.layer3_bottleneck2 = IdentityBlock(in_channels=l3ic[2], out_channels=l3oc[2], kernel_size=3, padding=1)
        self.layer3_bottleneck3 = IdentityBlock(in_channels=l3ic[3], out_channels=l3oc[3], kernel_size=3, padding=1)
        self.layer3_bottleneck4 = IdentityBlock(in_channels=l3ic[4], out_channels=l3oc[4], kernel_size=3, padding=1)
        self.layer3_bottleneck5 = IdentityBlock(in_channels=l3ic[5], out_channels=l3oc[5], kernel_size=3, padding=1)
        # layer4
        l4ic = [1024, 2048, 2048]
        l4oc = [2048, 2048, 2048]
        self.layer4_bottleneck0 = ConvBlock(    in_channels=l4ic[0], out_channels=l4oc[0], kernel_size=3, padding=2, dilation=2)
        self.layer4_bottleneck1 = IdentityBlock(in_channels=l4ic[1], out_channels=l4oc[1], kernel_size=3, padding=2, dilation=2)
        self.layer4_bottleneck2 = IdentityBlock(in_channels=l4ic[2], out_channels=l4oc[2], kernel_size=3, padding=2, dilation=2)
        #####
        # CSP specific layers
        #####
        # p3up
        self.p3up_trconv = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.p3up_l2norm = nn.L2Norm2d(256, 10)
        # p4up
        self.p4up_trconv = nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=4, padding=0)
        self.p4up_l2norm = nn.L2Norm2d(256, 10)
        # p5up
        self.p5up_trconv = nn.ConvTranspose2d(2048, 256, kernel_size=4, stride=4, padding=0)
        self.p5up_l2norm = nn.L2Norm2d(256, 10)
        # detection head - feat
        self.feat_conv = nn.Conv2d(768, 256, kernel_size=3, padding=1, bias=False)
        self.feat_bn = nn.BatchNorm2d(256)
        self.feat_relu = nn.ReLU(inplace=True)
        # detection head - class
        self.class_conv = nn.Conv2d(256, 1, kernel_size=1, bias=True)
        # self.class_sigmoid = nn.Sigmoid()
        # detection head - regr
        self.regr_conv = nn.Conv2d(256, 2, kernel_size=1)
        self.regr_relu = nn.ReLU(inplace=True)
        # detection head - offset
        self.offset_conv = nn.Conv2d(256, 2, kernel_size=1)
        self.offset_relu = nn.ReLU(inplace=True)

    def __call__(self, x):
        flops = int(0)
        # base
        out, flops_t = self.base_conv(x)
        flops += flops_t
        out, flops_t = self.base_bn(out)
        flops += flops_t
        out, flops_t = self.base_relu(out)
        flops += flops_t
        out, flops_t = self.base_maxpooling(out)
        flops += flops_t

        # layer1
        out, flops_t = self.layer1_bottleneck0(out)
        flops += flops_t
        out, flops_t = self.layer1_bottleneck1(out)
        flops += flops_t
        out, flops_t = self.layer1_bottleneck2(out)
        flops += flops_t

        # layer2
        out, flops_t = self.layer2_bottleneck0(out)
        flops += flops_t
        out, flops_t = self.layer2_bottleneck1(out)
        flops += flops_t
        out, flops_t = self.layer2_bottleneck2(out)
        flops += flops_t
        stage3, flops_t = self.layer2_bottleneck3(out)
        flops += flops_t

        # layer3
        out, flops_t = self.layer3_bottleneck0(stage3)
        flops += flops_t
        out, flops_t = self.layer3_bottleneck1(out)
        flops += flops_t
        out, flops_t = self.layer3_bottleneck2(out)
        flops += flops_t
        out, flops_t = self.layer3_bottleneck3(out)
        flops += flops_t
        out, flops_t = self.layer3_bottleneck4(out)
        flops += flops_t
        stage4, flops_t = self.layer3_bottleneck5(out)
        flops += flops_t

        # layer4
        out, flops_t = self.layer4_bottleneck0(stage4)
        flops += flops_t
        out, flops_t = self.layer4_bottleneck1(out)
        flops += flops_t
        stage5, flops_t = self.layer4_bottleneck2(out)
        flops += flops_t

        # pn_deconv
        p3up, flops_t = self.p3up_trconv(stage3)
        flops += flops_t
        p4up, flops_t = self.p4up_trconv(stage4)
        flops += flops_t
        p5up, flops_t = self.p5up_trconv(stage5)
        flops += flops_t

        # l2norm
        p3up, flops_t = self.p3up_l2norm(p3up)
        flops += flops_t
        p4up, flops_t = self.p4up_l2norm(p4up)
        flops += flops_t
        p5up, flops_t = self.p5up_l2norm(p5up)
        flops += flops_t

        # concat
        conc = flops_counter.cat((p3up, p4up, p5up), 1)

        # detection head - feat
        feat, flops_t = self.feat_conv(conc)
        flops += flops_t
        feat, flops_t = self.feat_bn(feat)
        flops += flops_t
        feat, flops_t = self.feat_relu(feat)
        flops += flops_t

        # detection head - class
        x_class, flops_t = self.class_conv(feat)
        flops += flops_t
        # x_class, flops_t = self.class_sigmoid(x_class)
        # flops += flops_t

        # detection head - regr
        x_regr, flops_t = self.regr_conv(feat)
        flops += flops_t
        x_regr, flops_t = self.regr_relu(x_regr)
        flops += flops_t

        # detection head - offset
        x_offset, flops_t = self.offset_conv(feat)
        flops += flops_t
        x_offset, flops_t = self.offset_relu(x_offset)
        flops += flops_t

        return flops