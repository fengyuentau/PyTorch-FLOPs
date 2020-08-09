import flops_counter
import flops_counter.nn as nn
import flops_counter.nn.functional as F

class SHF(nn.Module):
    def __init__(self):
        super(SHF, self).__init__()

        self.conv1 = self.build_conv_block(3,    64, with_pool=True)
        self.conv2 = self.build_conv_block(64,  128, with_pool=True)
        self.conv3 = self.build_conv_block(128, 256, n_conv=3, with_pool=True)
        self.conv4 = self.build_conv_block(256, 512, n_conv=3)

        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5 = self.build_conv_block(512, 512, n_conv=3)
        self.conv5_256 = self.build_conv_block(512, 256, 1, padding=0, n_conv=1)
        self.conv5_256_up = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, groups=256, bias=False)

        self.conv4_256 = self.build_conv_block(512, 256, 1, padding=0, n_conv=1)

        self.conv4_fuse_final = self.build_conv_block(512, 512, n_conv=1)
        self.conv4_fuse_final_dim_red = self.build_conv_block(512, 128, n_conv=1)

        self.head1 = self.build_conv_block(128, 128, padding=1, dilation=1, n_conv=1)
        self.head2 = self.build_conv_block(128, 128, padding=2, dilation=2, n_conv=1)
        self.head4 = self.build_conv_block(128, 128, padding=4, dilation=4, n_conv=1)

        self.bbox_head, self.cls_head = self.build_det_head()
        self.softmax = nn.Softmax(dim=1)

    def build_conv_block(self,
                         in_channels:   int,
                         out_channels:  int,
                         kernel_size:   int = 3,
                         stride:        int = 1,
                         padding:       int = 1,
                         dilation:      int = 1,
                         n_conv:        int = 2,
                         with_pool:     bool = False):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation),
            nn.ReLU()
        ]
        for i in range(1, n_conv):
            layers += [
                nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
                nn.ReLU()
            ]
        if with_pool:
            layers += [
                nn.MaxPool2d(2, 2)
            ]
        return nn.Sequential(*layers)

    def build_det_head(self):
        bbox_head = []
        cls_head = []
        for i in range(3):
            bbox_head += [ nn.Conv2d(128, 4, kernel_size=1, stride=1, padding=0) ]
            cls_head += [ nn.Conv2d(128, 2, kernel_size=1, stride=1, padding=0) ]
        return nn.ModuleList(bbox_head), nn.ModuleList(cls_head)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        conv4_3 = self.conv4(x)

        # branch 1
        conv5_3 = self.conv5(self.pool4(conv4_3))
        conv5_256 = self.conv5_256(conv5_3)
        conv5_256_up = self.conv5_256_up(conv5_256)
        # branch 2
        conv4_256 = self.conv4_256(conv4_3)
        # fuse(conv5_256_up, conv4_256)
        if conv5_256_up.value[2] != conv4_256.value[3] or conv5_256_up.value[3] != conv4_256.value[3]:
            pad = (0, conv4_256.value[3] - conv5_256_up.value[3], 0, conv4_256.value[2] - conv5_256_up.value[2])
            conv5_256_up = F.pad(conv5_256_up, pad)
        conv4_fuse = flops_counter.cat([conv5_256_up, conv4_256], 1)

        conv4_fuse_final_dim_red = self.conv4_fuse_final_dim_red(self.conv4_fuse_final(conv4_fuse))

        head1_f = self.head1(conv4_fuse_final_dim_red)
        head2_f = self.head2(conv4_fuse_final_dim_red)
        head4_f = self.head4(conv4_fuse_final_dim_red)
        outs = [head1_f, head2_f, head4_f]

        loc = []
        conf = []
        for i, o in enumerate(outs):
            loc += [ self.bbox_head[i](o) ]
            conf += [ self.cls_head[i](o) ]

        loc_cat = flops_counter.cat(loc, 1)
        conf_cat = self.softmax(flops_counter.cat(conf, 2))

        return loc_cat, conf_cat

    @property
    def name(self):
        return self._get_name()