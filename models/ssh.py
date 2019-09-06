import flops_counter
import flops_counter.nn as nn
import flops_counter.nn.functional as F

class M_Module(nn.Module):
    def __init__(self, in_channels, out_channels_left, out_channels_right):
        super(M_Module, self).__init__()

        inc = in_channels
        ocl, ocr = out_channels_left, out_channels_right

        # left branch
        self.ssh_3x3 = nn.Conv2d(inc, ocl, 3, 1, 1)
        # right branch
        self.ssh_dimred = nn.Conv2d(inc, ocr, 3, 1, 1)
        self.ssh_dimred_relu = nn.ReLU(inplace=True)
        self.ssh_5x5 = nn.Conv2d(ocr, ocr, 3, 1, 1)
        self.ssh_7x7_1 = nn.Conv2d(ocr, ocr, 3, 1, 1)
        self.ssh_7x7_1_relu = nn.ReLU(inplace=True)
        self.ssh_7x7 = nn.Conv2d(ocr, ocr, 3, 1, 1)

        self.ssh_output_relu = nn.ReLU(inplace=True)


    def forward(self, x):
        ssh_3x3 = self.ssh_3x3(x)

        ssh_dimred = self.ssh_dimred(x)
        ssh_dimred_relu = self.ssh_dimred_relu(ssh_dimred)
        ssh_5x5 = self.ssh_5x5(ssh_dimred_relu)

        ssh_7x7_1 = self.ssh_7x7_1(ssh_dimred_relu)
        ssh_7x7_1_relu = self.ssh_7x7_1_relu(ssh_7x7_1)
        ssh_7x7 = self.ssh_7x7(ssh_7x7_1_relu)

        # merge
        ssh_output = flops_counter.cat([ssh_3x3, ssh_5x5, ssh_7x7], 1)
        ssh_output_relu = self.ssh_output_relu(ssh_output)

        return ssh_output_relu

class SSH(nn.Module):
    def __init__(self):
        super(SSH, self).__init__()

        # backbone
        self.vgg16 = nn.ModuleList(make_layers(vgg_cfgs['D']))

        # SSH - M3
        self.M3 = M_Module(512, 256, 128)
        self.M3_bbox_pred = nn.Conv2d(512, 8, 1, 1, 0)
        self.M3_cls_score = nn.Conv2d(512, 4, 1, 1, 0)
        self.M3_cls_score_softmax = nn.Softmax(dim=1)
        # SSH - M2
        self.M2 = M_Module(512, 256, 128)
        self.M2_bbox_pred = nn.Conv2d(512, 8, 1, 1, 0)
        self.M2_cls_score = nn.Conv2d(512, 4, 1, 1, 0)
        self.M2_cls_score_softmax = nn.Softmax(dim=1)
        # SSH - M1
        self.conv4_128 = nn.Conv2d(512, 128, 1, 1, 0)
        self.conv4_128_relu = nn.ReLU(inplace=True)
        self.conv5_128 = nn.Conv2d(512, 128, 1, 1, 0)
        self.conv5_128_relu = nn.ReLU(inplace=True)
        self.conv5_128_up = nn.ConvTranspose2d(128, 128, 4, 2, 1, groups=128, bias=False)
        self.eltadd = nn.EltAdd()
        self.conv4_fuse_final = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv4_fuse_final_relu = nn.ReLU(inplace=True)
        self.M1 = M_Module(128, 128, 64)
        self.M1_bbox_pred = nn.Conv2d(256, 8, 1, 1, 0)
        self.M1_cls_score = nn.Conv2d(256, 4, 1, 1, 0)
        self.M1_cls_score_softmax = nn.Softmax(dim=1)

    @property
    def name(self):
        return self._get_name() + '_VGG16'

    def forward(self, x):
        out = x

        # get conv4_3
        for k in range(23):
            out = self.vgg16[k](out)
        conv4_3 = out

        # get conv5_3
        for k in range(23, 30):
            out = self.vgg16[k](out)
        conv5_3 = out

        # get pool6 (it's actually pool5)
        pool6 = self.vgg16[30](out)

        self.vgg16.settle(x, pool6)

        # SSH - M3
        M3_output = self.M3(pool6)
        # SSH - M3 - bbox_pred
        M3_bbox_pred = self.M3_bbox_pred(M3_output)
        # SSH - M3 - cls_score
        M3_cls_score = self.M3_cls_score(M3_output)
        M3_cls_score = M3_cls_score.view(M3_cls_score.value[0], 2, -1, M3_cls_score.value[-1])
        M3_cls_prob = self.M3_cls_score_softmax(M3_cls_score)
        M3_cls_prob = M3_cls_prob.view(M3_cls_prob.value[0], 4, -1, M3_cls_prob.value[-1])

        # SSH - M2
        M2_output = self.M2(conv5_3)
        # SSH - M2 - bbox_pred
        M2_bbox_pred = self.M2_bbox_pred(M2_output)
        # SSH - M2 - cls_score
        M2_cls_score = self.M2_cls_score(M2_output)
        M2_cls_score = M2_cls_score.view(M2_cls_score.value[0], 2, -1, M2_cls_score.value[-1])
        M2_cls_prob = self.M2_cls_score_softmax(M2_cls_score)
        M2_cls_prob = M2_cls_prob.view(M2_cls_prob.value[0], 4, -1, M2_cls_prob.value[-1])

        # SSH - M1
        conv4_128 = self.conv4_128(conv4_3)
        conv4_128 = self.conv4_128_relu(conv4_128)
        conv5_128 = self.conv5_128(conv5_3)
        conv5_128 = self.conv5_128_relu(conv5_128)
        conv5_128_up = self.conv5_128_up(conv5_128)
        if conv5_128_up.value[2] != conv4_128.value[2] or conv5_128_up.value[3] != conv4_128.value[3]:
            pads = (0, conv4_128.value[3] - conv5_128_up.value[3], 0, conv4_128.value[2] - conv5_128_up.value[2])
            conv5_128_up = F.pad(conv5_128_up, pads)
        conv4_fuse = self.eltadd(conv4_128, conv5_128_up)
        conv4_fuse_final = self.conv4_fuse_final(conv4_fuse)
        conv4_fuse_final = self.conv4_fuse_final_relu(conv4_fuse_final)
        M1_output = self.M1(conv4_fuse_final)
        # SSH - M2 - bbox_pred
        M1_bbox_pred = self.M1_bbox_pred(M1_output)
        # SSH - M2 - cls_score
        M1_cls_score = self.M1_cls_score(M1_output)
        M1_cls_score = M1_cls_score.view(M1_cls_score.value[0], 2, -1, M1_cls_score.value[-1])
        M1_cls_prob = self.M1_cls_score_softmax(M1_cls_score)
        M1_cls_prob = M1_cls_prob.view(M1_cls_prob.value[0], 4, -1, M1_cls_prob.value[-1])

        return (M1_bbox_pred, M2_bbox_pred, M3_bbox_pred), (M1_cls_prob, M2_cls_prob, M3_cls_prob)






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