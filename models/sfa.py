import flops_counter
import flops_counter.nn as nn
import flops_counter.nn.functional as F

class SFA(nn.Module):
    def __init__(self):
        super(SFA, self).__init__()

        #                        M0,  M1,  M2,  M3
        self.ssh_in_channels  = [128, 128, 512, 512]
        self.ssh_out_channels = [256, 256, 512, 512]

        self.conv1 = self.build_conv_block(in_channels=3,   out_channels=64,  n_conv=2, with_pool=False)
        self.conv2 = self.build_conv_block(in_channels=64,  out_channels=128, n_conv=2, with_pool=True)
        self.conv3 = self.build_conv_block(in_channels=128, out_channels=256, n_conv=3, with_pool=True)
        self.conv4 = self.build_conv_block(in_channels=256, out_channels=512, n_conv=3, with_pool=True)
        self.conv5 = self.build_conv_block(in_channels=512, out_channels=512, n_conv=3, with_pool=True)

        # M3
        self.pool6 = nn.MaxPool2d(2)
        self.m3 = SSH(self.ssh_in_channels[3], self.ssh_out_channels[3], index=3)

        # M2
        self.m2 = SSH(self.ssh_in_channels[2], self.ssh_out_channels[2], index=2)

        # share by M1 and M2
        self.conv4_128 = self.build_conv_block(in_channels=512, out_channels=128, kernel_size=1, padding=0, n_conv=1, with_pool=False)

        # M1
        self.conv5_128 = self.build_conv_block(in_channels=512, out_channels=128, kernel_size=1, padding=0, n_conv=1, with_pool=False)
        self.conv5_128_up = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1, groups=128, bias=False)
        self.conv4_fuse = nn.EltAdd()
        self.conv4_fuse_final = self.build_conv_block(in_channels=128, out_channels=128, kernel_size=3, padding=1, n_conv=1, with_pool=False)
        self.m1 = SSH(self.ssh_in_channels[1], self.ssh_out_channels[1], index=1)

        # M0
        self.conv3_128 = self.build_conv_block(in_channels=256, out_channels=128, kernel_size=1, padding=0, n_conv=1, with_pool=False)
        self.conv4_128_up = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1, groups=128, bias=False)
        self.conv3_fuse = nn.EltAdd()
        self.conv3_fuse_final = self.build_conv_block(in_channels=128, out_channels=128, kernel_size=3, padding=1, n_conv=1, with_pool=False)
        self.m0 = SSH(self.ssh_in_channels[0], self.ssh_out_channels[0], index=0)

        # detection heads
        self.bbox_head, self.cls_head = self.build_detect_head()
        self.softmax = nn.Softmax(dim=-1)

    def build_conv_block(self,
                         in_channels:  int,
                         out_channels: int,
                         kernel_size:  int = 3,
                         stride:       int = 1,
                         padding:      int = 1,
                         n_conv:       int = 2,
                         with_pool:    bool = False):
        layers = []

        if with_pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # convx_1
        layers += [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU()
        ]
        # convx_2 -> convx_(n_conv)
        for i in range(1, n_conv):
            add_layers = [
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.ReLU()
            ]
            layers += add_layers

        # return as sequential
        return nn.Sequential(*layers)

    def build_detect_head(self):
        bbox_pred = []
        cls_score = []
        for oc in self.ssh_out_channels:
            bbox_pred += [ nn.Conv2d(in_channels=oc, out_channels=8, kernel_size=1, stride=1, padding=0) ]
            cls_score += [ nn.Conv2d(in_channels=oc, out_channels=4, kernel_size=1, stride=1, padding=0) ]

        return nn.ModuleList(bbox_pred), nn.ModuleList(cls_score)


    def forward(self, x):
        outs = []

        conv3_3 = self.conv3(self.conv2(self.conv1(x)))
        conv4_3 = self.conv4(conv3_3)
        conv5_3 = self.conv5(conv4_3)

        # M3
        pool6 = self.pool6(conv5_3)
        m3_out = self.m3(pool6)
        outs.append(m3_out)

        # M2
        m2_out = self.m2(conv5_3)
        outs.append(m2_out)

        # share by M1 and M0
        conv4_128 = self.conv4_128(conv4_3)

        # M1
        conv5_128 = self.conv5_128(conv5_3)
        conv5_128_up = self.conv5_128_up(conv5_128)
        #  padding so that conv5_128_up can have the same size as conv4_128
        if conv4_128.value[2] != conv5_128_up.value[2] or conv4_128.value[3] != conv5_128_up.value[3]:
            pad = (0, conv5_128_up.value[3] - conv4_128.value[3], 0, conv5_128_up.value[2] - conv4_128.value[2])
            conv5_128_up = F.pad(conv5_128_up, pad)
        conv4_fuse = self.conv4_fuse(conv4_128, conv5_128_up)
        conv4_fuse_final = self.conv4_fuse_final(conv4_fuse)
        m1_out = self.m1(conv4_fuse_final)
        outs.append(m1_out)

        # M0
        conv3_128 = self.conv3_128(conv3_3)
        conv4_128_up = self.conv4_128_up(conv4_128)
        #  padding so that conv4_128_up can have the same size as conv3_128
        if conv3_128.value[2] != conv4_128_up.value[2] or conv3_128.value[3] != conv4_128_up.value[3]:
            pad = (0, conv4_128_up.value[3] - conv3_128.value[3], 0, conv4_128_up.value[2] - conv3_128.value[2])
            conv4_128_up = F.pad(conv4_128_up, pad)
        conv3_fuse = self.conv3_fuse(conv3_128, conv4_128_up)
        conv3_fuse_final = self.conv3_fuse_final(conv3_fuse)
        m0_out = self.m0(conv3_fuse_final)
        outs.append(m0_out)

        # detection head
        loc = []
        conf = []
        for i, o in enumerate(outs[::-1]): # reverse outs so that the order of feature maps match with the order of heads
            loc.append(self.bbox_head[i](o))

            cls_score = self.cls_head[i](o)
            cls_score = cls_score.view(cls_score.size(0), -1, 4)
            cls_score = self.softmax(cls_score)
            conf.append(cls_score)

        return loc, conf

    @property
    def name(self):
        return self._get_name()

class SSH(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, index: int):
        super(SSH, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.half_out_channels = int(out_channels / 2)
        self.quater_out_channels = int(self.half_out_channels / 2)
        self.index = index
        
        self.ssh_3x3 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.half_out_channels, kernel_size=3, stride=1, padding=1)
        )

        self.ssh_dimred = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.quater_out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.ssh_5x5 = nn.Sequential(
            nn.Conv2d(in_channels=self.quater_out_channels, out_channels=self.quater_out_channels, kernel_size=3, stride=1, padding=1)
        )

        self.ssh_7x7 = nn.Sequential(
            nn.Conv2d(in_channels=self.quater_out_channels, out_channels=self.quater_out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.quater_out_channels, out_channels=self.quater_out_channels, kernel_size=3, stride=1, padding=1)
        )

        self.out_relu = nn.ReLU()

    def forward(self, x):
        ssh_3x3_f = self.ssh_3x3(x)

        ssh_dimred_f = self.ssh_dimred(x)

        ssh_5x5_f = self.ssh_5x5(ssh_dimred_f)

        ssh_7x7_f = self.ssh_7x7(ssh_dimred_f)

        return self.out_relu(flops_counter.cat([ssh_3x3_f, ssh_5x5_f, ssh_7x7_f], 1))

    @property
    def name(self):
        return self._get_name() + '-' + str(self.index)