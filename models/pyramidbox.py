import six
import numpy as np

import flops_counter
import flops_counter.nn.functional as F
import flops_counter.nn as nn

class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, strides=None, paddings=None, with_pool=True):
        super(Conv_Block, self).__init__()
        assert len(in_channels) == len(out_channels)
        assert len(out_channels) == len(kernel_sizes)
        if strides is not None:
            assert len(kernel_sizes) == len(strides)

        self.pool = None
        if with_pool:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        groups = len(in_channels)
        convs = []
        for i in range(groups):
            convs.append(nn.Conv2d(in_channels=in_channels[i], out_channels=out_channels[i], kernel_size=kernel_sizes[i], stride=strides[i], padding=paddings[i]))
            convs.append(nn.ReLU(inplace=True))
        self.feature = nn.Sequential(*convs)

    def forward(self, x):
        out = self.feature(x)
        if self.pool:
            pool = self.pool(out)
            return out, pool
        return out

class Conv_BN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, strides=1, paddings=0, act='relu', bias=False):
        super(Conv_BN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_sizes, strides, paddings, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.999)
        self.act = None
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        if self.act:
            out = self.act(out)
        return out


class LFPN(nn.Module):
    def __init__(self, up_from_channels, up_to_channels):
        super(LFPN, self).__init__()

        self.conv1 = nn.Conv2d(up_from_channels, up_to_channels, kernel_size=1)
        self.conv1_relu = nn.ReLU(inplace=True)

        self.upsampling = nn.ConvTranspose2d(
            up_to_channels,
            up_to_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            groups=up_to_channels,
            bias=False
        )

        self.conv2 = nn.Conv2d(up_to_channels, up_to_channels, kernel_size=1)
        self.conv2_relu = nn.ReLU(inplace=True)
        self.eltmul = nn.EltMul()

    def forward(self, up_from, up_to):
        conv1 = self.conv1(up_from)
        conv1 = self.conv1_relu(conv1)
        upsampling = self.upsampling(conv1)

        conv2 = self.conv2(up_to)
        conv2 = self.conv2_relu(up_to)

        if upsampling.value[2] != conv2.value[2] or upsampling.value[3] != conv2.value[3]:
            # upsampling = upsampling[:, :, 0:conv2.size(2), 0:conv2.size(3)]
            pads = (0, conv2.value[3] - upsampling.value[3], 0, conv2.value[2] - upsampling.value[2])
            upsampling = F.pad(upsampling, pads)

        fuse = self.eltmul(upsampling, conv2)
        return fuse

class CPM(nn.Module):
    def __init__(self, in_channels):
        super(CPM, self).__init__()
        # residual
        self.branch1 = Conv_BN(in_channels, 1024, 1, 1, 0, act=None)
        self.branch2a = Conv_BN(in_channels, 256, 1, 1, 0, act='relu')
        self.branch2b = Conv_BN(256, 256, 3, 1, 1, act='relu')
        self.branch2c = Conv_BN(256, 1024, 1, 1, 0, act=None)
        self.eltadd = nn.EltAdd()
        self.rescomb_relu = nn.ReLU(inplace=True)

        # ssh
        self.ssh_1_conv = nn.Conv2d(1024, 256, 3, 1, 1)
        self.ssh_dimred_conv = nn.Conv2d(1024, 128, 3, 1, 1)
        self.ssh_dimred_relu = nn.ReLU(inplace=True)
        self.ssh_2_conv = nn.Conv2d(128, 128, 3, 1, 1)
        self.ssh_3a_conv = nn.Conv2d(128, 128, 3, 1, 1)
        self.ssh_3a_relu = nn.ReLU(inplace=True)
        self.ssh_3b_conv = nn.Conv2d(128, 128, 3, 1, 1)

        self.concat_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # residual
        branch1 = self.branch1(x)
        branch2a = self.branch2a(x)
        branch2b = self.branch2b(branch2a)
        branch2c = self.branch2c(branch2b)
        branch_sum = self.eltadd(branch1, branch2c)
        rescomb = self.rescomb_relu(branch_sum)

        # ssh
        ssh_1 = self.ssh_1_conv(rescomb)
        ssh_dimred = self.ssh_dimred_conv(rescomb)
        ssh_dimred = self.ssh_dimred_relu(ssh_dimred)
        ssh_2 = self.ssh_2_conv(ssh_dimred)
        ssh_3a = self.ssh_3a_conv(ssh_dimred)
        ssh_3a = self.ssh_3a_relu(ssh_3a)
        ssh_3b = self.ssh_3b_conv(ssh_3a)

        ssh_concat = flops_counter.cat([ssh_1, ssh_2, ssh_3b], 1)
        ssh_concat = self.concat_relu(ssh_concat)

        return ssh_concat

class PyramidBox(nn.Module):
    def __init__(self):
        super(PyramidBox, self).__init__()

        self.img_shape = [3, 1024, 1024]

        self._vgg()
        self._low_level_fpn()
        self._cpm_module()
        self._pyramidbox()

    @property
    def name(self):
        return self._get_name() + '_VGG16'

    def _vgg(self):
        in_channels = [3, 64]
        out_channels = [64, 64]
        kernel_sizes = [3, 3]
        strides = [1, 1]
        paddings = [((ks - 1) // 2) for ks in kernel_sizes]
        self.layer1 = Conv_Block(in_channels, out_channels, kernel_sizes, strides, paddings)

        in_channels = [64, 128]
        out_channels = [128, 128]
        kernel_sizes = [3, 3]
        strides = [1, 1]
        paddings = [((ks - 1) // 2) for ks in kernel_sizes]
        self.layer2 = Conv_Block(in_channels, out_channels, kernel_sizes, strides, paddings)

        in_channels = [128, 256, 256]
        out_channels = [256, 256, 256]
        kernel_sizes = [3, 3, 3]
        strides = [1, 1, 1]
        paddings = [((ks - 1) // 2) for ks in kernel_sizes]
        self.layer3 = Conv_Block(in_channels, out_channels, kernel_sizes, strides, paddings)

        in_channels = [256, 512, 512]
        out_channels = [512, 512, 512]
        kernel_sizes = [3, 3, 3]
        strides = [1, 1, 1]
        paddings = [((ks - 1) // 2) for ks in kernel_sizes]
        self.layer4 = Conv_Block(in_channels, out_channels, kernel_sizes, strides, paddings)

        in_channels = [512, 512, 512]
        out_channels = [512, 512, 512]
        kernel_sizes = [3, 3, 3]
        strides = [1, 1, 1]
        paddings = [((ks - 1) // 2) for ks in kernel_sizes]
        self.layer5 = Conv_Block(in_channels, out_channels, kernel_sizes, strides, paddings)

        in_channels = [512, 1024]
        out_channels = [1024, 1024]
        kernel_sizes = [3, 1]
        strides = [1, 1]
        paddings = [((ks - 1) // 2) for ks in kernel_sizes]
        self.layer6 = Conv_Block(in_channels, out_channels, kernel_sizes, strides, paddings, with_pool=False)

        in_channels = [1024, 256]
        out_channels = [256, 512]
        kernel_sizes = [1, 3]
        strides = [1, 2]
        paddings = [((ks - 1) // 2) for ks in kernel_sizes]
        self.layer7 = Conv_Block(in_channels, out_channels, kernel_sizes, strides, paddings, with_pool=False)

        in_channels = [512, 128]
        out_channels = [128, 256]
        kernel_sizes = [1, 3]
        strides = [1, 2]
        paddings = [((ks - 1) // 2) for ks in kernel_sizes]
        self.layer8 = Conv_Block(in_channels, out_channels, kernel_sizes, strides, paddings, with_pool=False)

    def _low_level_fpn(self):
        self.lfpn2_on_conv5 = LFPN(1024, 512)
        self.lfpn1_on_conv4 = LFPN(512, 512)
        self.lfpn0_on_conv3 = LFPN(512, 256)

    def _cpm_module(self):
        self.ssh_conv3 = CPM(256)
        self.ssh_conv4 = CPM(512)
        self.ssh_conv5 = CPM(512)
        self.ssh_conv6 = CPM(1024)
        self.ssh_conv7 = CPM(512)
        self.ssh_conv8 = CPM(256)

    def _pyramidbox(self):
        self.ssh_conv3_l2norm = nn.L2Norm2d(512, 10)
        self.ssh_conv4_l2norm = nn.L2Norm2d(512, 8)
        self.ssh_conv5_l2norm = nn.L2Norm2d(512, 5)

        self.SSHchannels = [512,512,512,512,512,512]
        loc = []
        conf = []
        for i in range(6):
            loc.append(nn.Conv2d(self.SSHchannels[i], 8, kernel_size=3, stride=1, padding=1))
            if i == 0:
                conf.append(nn.Conv2d(self.SSHchannels[i], 8, kernel_size=3, stride=1, padding=1))
            else:
                conf.append(nn.Conv2d(self.SSHchannels[i], 6, kernel_size=3, stride=1, padding=1))
        self.mbox_loc = nn.ModuleList(loc)
        self.mbox_conf = nn.ModuleList(conf)

        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x):
        conv1, pool1 = self.layer1(x)
        conv2, pool2 = self.layer2(pool1)

        conv3, pool3 = self.layer3(pool2)
        conv4, pool4 = self.layer4(pool3)
        conv5, pool5 = self.layer5(pool4)

        conv6 = self.layer6(pool5)
        conv7 = self.layer7(conv6)
        conv8 = self.layer8(conv7)

        lfpn2_on_conv5 = self.lfpn2_on_conv5(conv6, conv5)
        lfpn1_on_conv4 = self.lfpn1_on_conv4(lfpn2_on_conv5, conv4)
        lfpn0_on_conv3 = self.lfpn0_on_conv3(lfpn1_on_conv4, conv3)

        ssh_conv3 = self.ssh_conv3(lfpn0_on_conv3)
        ssh_conv4 = self.ssh_conv4(lfpn1_on_conv4)
        ssh_conv5 = self.ssh_conv5(lfpn2_on_conv5)
        ssh_conv6 = self.ssh_conv6(conv6)
        ssh_conv7 = self.ssh_conv7(conv7)
        ssh_conv8 = self.ssh_conv8(conv8)

        ssh_conv3_l2norm = self.ssh_conv3_l2norm(ssh_conv3)
        ssh_conv4_l2norm = self.ssh_conv4_l2norm(ssh_conv4)
        ssh_conv5_l2norm = self.ssh_conv5_l2norm(ssh_conv5)

        inputs = [ssh_conv3_l2norm, ssh_conv4_l2norm, ssh_conv5_l2norm, ssh_conv6, ssh_conv7, ssh_conv8]
        face_confs = []
        head_confs = []
        for i, feat in enumerate(inputs):
            mbox_loc = self.mbox_loc[i](feat)
            # print(mbox_loc)
            if i == 0:
                temp_conf = self.mbox_conf[i](feat)
                # face_conf3 = temp_conf[:, 0:3, :, :]
                face_conf3 = [x for x in temp_conf.value]
                face_conf3[1] = 3
                face_conf3 = flops_counter.TensorSize(face_conf3)
                # face_conf1 = temp_conf[:, 3:4, :, :]
                face_conf1 = [x for x in temp_conf.value]
                face_conf1[1] = 1
                face_conf1 = flops_counter.TensorSize(face_conf1)
                # head_conf3 = temp_conf[:, 4:7, :, :]
                head_conf3 = [x for x in temp_conf.value]
                head_conf3[1] = 3
                head_conf3 = flops_counter.TensorSize(head_conf3)
                # head_conf1 = temp_conf[:, 7:, :, :]
                head_conf1 = [x for x in temp_conf.value]
                head_conf1[1] = 1
                head_conf1 = flops_counter.TensorSize(head_conf1)
                # face conf
                face_conf3_maxin = face_conf3.max(1, keepdim=True)
                face_confs.append(flops_counter.cat([face_conf3_maxin, face_conf1], 1).permute(0, 2, 3, 1))
                # head conf
                head_conf3_maxin = head_conf3.max(1, keepdim=True)
                head_confs.append(flops_counter.cat([head_conf3_maxin, head_conf1], 1).permute(0, 2, 3, 1))
            else:
                temp_conf = self.mbox_conf[i](feat)
                # face_conf1 = temp_conf[:, 0:1, :, :]
                face_conf1 = [x for x in temp_conf.value]
                face_conf1[1] = 1
                face_conf1 = flops_counter.TensorSize(face_conf1)
                # face_conf3 = temp_conf[:, 1:4, :, :]
                face_conf3 = [x for x in temp_conf.value]
                face_conf3[1] = 3
                face_conf3 = flops_counter.TensorSize(face_conf3)
                # head_conf = temp_conf[:, 4:, :, :]
                head_conf1 = [x for x in temp_conf.value]
                head_conf1[1] = 4
                head_conf1 = flops_counter.TensorSize(head_conf1)

                # face conf
                face_conf3_maxin = face_conf3.max(1, keepdim=True)
                face_confs.append(flops_counter.cat([face_conf1, face_conf3_maxin], 1).permute(0, 2, 3, 1))
                # head conf
                head_confs.append(head_conf1.permute(0, 2, 3, 1))
            # print(temp_conf)

        face_conf = flops_counter.cat([o.view(o.value[0], -1) for o in face_confs], 1)

        head_conf = flops_counter.cat([o.view(o.value[0], -1) for o in head_confs], 1)

        face_conf_softmax = self.softmax(face_conf.view(face_conf.value[0], -1, 2))

        return face_conf_softmax