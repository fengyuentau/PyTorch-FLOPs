import flops_counter
import flops_counter.nn as nn
from vision import models

class S3FD(nn.Module):
    def __init__(self):
        super(S3FD, self).__init__()

        # backbone
        self.vgg16 = nn.ModuleList(make_layers(vgg_cfgs['D']))

        # s3fd specific
        self.conv_fc6 = nn.Conv2d(512, 1024, 3, 1, 1)
        self.relu_fc6 = nn.ReLU()
        self.conv_fc7 = nn.Conv2d(1024, 1024, 1, 1, 0)
        self.relu_fc7 = nn.ReLU()

        self.conv6_1 = nn.Conv2d(1024, 256, 1, 1, 0)
        self.relu_conv6_1 = nn.ReLU()
        self.conv6_2 = nn.Conv2d(256, 512, 3, 2, 1)
        self.relu_conv6_2 = nn.ReLU()

        self.conv7_1 = nn.Conv2d(512, 128, 1, 1, 0)
        self.relu_conv7_1 = nn.ReLU()
        self.conv7_2 = nn.Conv2d(128, 256, 3, 2, 1)
        self.relu_conv7_2 = nn.ReLU()

        self.l2norm_conv3_3 = nn.L2Norm2d(256, 10)
        self.l2norm_conv4_3 = nn.L2Norm2d(512, 8)
        self.l2norm_conv5_3 = nn.L2Norm2d(512, 5)

        # Detection Head - mbox_loc
        self.mbox_loc_conv3_3_norm  = nn.Conv2d(256, 4, 3, 1, 1)
        self.mbox_loc_conv4_3_norm  = nn.Conv2d(512, 4, 3, 1, 1)
        self.mbox_loc_conv5_3_norm  = nn.Conv2d(512, 4, 3, 1, 1)
        self.mbox_loc_conv_fc7      = nn.Conv2d(1024, 4, 3, 1, 1)
        self.mbox_loc_conv6_2       = nn.Conv2d(512, 4, 3, 1, 1)
        self.mbox_loc_conv7_2       = nn.Conv2d(256, 4, 3, 1, 1)
        # Detection Head - mbox_conf
        self.mbox_conf_conv3_3_norm = nn.Conv2d(256, 4, 3, 1, 1) # 4->2 through maxout at channels 0~2
        self.mbox_conf_conv4_3_norm = nn.Conv2d(512, 2, 3, 1, 1)
        self.mbox_conf_conv5_3_norm = nn.Conv2d(512, 2, 3, 1, 1)
        self.mbox_conf_conv_fc7     = nn.Conv2d(1024, 2, 3, 1, 1)
        self.mbox_conf_conv6_2      = nn.Conv2d(512, 2, 3, 1, 1)
        self.mbox_conf_conv7_2      = nn.Conv2d(256, 2, 3, 1, 1)
        # Detection Head - mbox_conf - softmax
        self.softmax = nn.Softmax(dim=-1)


    @property
    def name(self):
        return self._get_name() + '_VGG16'

    def forward(self, x):
        out = x

        # get conv3_3
        for k in range(16):
            out = self.vgg16[k](out)
        conv3_3 = out # channels = 256
        conv3_3_norm = self.l2norm_conv3_3(conv3_3)

        # get conv4_3
        for k in range(16, 23):
            out = self.vgg16[k](out)
        conv4_3 = out # channels = 512
        conv4_3_norm = self.l2norm_conv4_3(conv4_3)

        # get conv5_3
        for k in range(23, 30):
            out = self.vgg16[k](out)
        conv5_3 = out # channels = 512
        conv5_3_norm = self.l2norm_conv5_3(conv5_3)

        out = self.vgg16[30](out)

        # get conv_fc7
        out = self.conv_fc6(out)
        out = self.relu_fc6(out)
        out = self.conv_fc7(out)
        out = self.relu_fc7(out)
        conv_fc7 = out

        # get conv6_2
        out = self.conv6_1(out)
        out = self.relu_conv6_1(out)
        out = self.conv6_2(out)
        out = self.relu_conv6_2(out)
        conv6_2 = out

        # get conv7_2
        out = self.conv7_1(out)
        out = self.relu_conv7_1(out)
        out = self.conv7_2(out)
        out = self.relu_conv7_2(out)
        conv7_2 = out

        # Detection Head - mbox_loc
        mbox_loc_inputs = [
            self.mbox_loc_conv3_3_norm(conv3_3_norm),
            self.mbox_loc_conv4_3_norm(conv4_3_norm),
            self.mbox_loc_conv5_3_norm(conv5_3_norm),
            self.mbox_loc_conv_fc7(conv_fc7),
            self.mbox_loc_conv6_2(conv6_2),
            self.mbox_loc_conv7_2(conv7_2)
        ]
        mbox_loc = flops_counter.cat([o.permute(0, 2, 3, 1).view(1, -1, 4) for o in mbox_loc_inputs], 1)
        # Detection Head - mbox_conf
        mbox_conf_conv3_3_norm = self.mbox_conf_conv3_3_norm(conv3_3_norm)
        
        conf1 = [i for i in mbox_conf_conv3_3_norm.value]
        conf1[1] = 1
        conf1 = flops_counter.TensorSize(conf1)
        
        conf234 = [i for i in mbox_conf_conv3_3_norm.value]
        conf234[1] = 3
        conf234 = flops_counter.TensorSize(conf234)
        conf234 = conf234.max(1, keepdim=True)
        
        mbox_conf_conv3_3_norm = flops_counter.cat([conf1, conf234], 1)

        mbox_conf_inputs = [
            mbox_conf_conv3_3_norm,
            self.mbox_conf_conv4_3_norm(conv4_3_norm),
            self.mbox_conf_conv5_3_norm(conv5_3_norm),
            self.mbox_conf_conv_fc7(conv_fc7),
            self.mbox_conf_conv6_2(conv6_2),
            self.mbox_conf_conv7_2(conv7_2)
        ]
        mbox_conf = flops_counter.cat([o.permute(0, 2, 3, 1).view(1, -1, 2) for o in mbox_conf_inputs], 1)
        mbox_conf = self.softmax(mbox_conf)

        return mbox_loc, mbox_conf



vgg_cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return layers