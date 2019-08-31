import flops_counter
import flops_counter.nn.functional as F
import flops_counter.nn as nn
import vision

class FEM(nn.Module):
    def __init__(self, channel_size):
        super(FEM, self).__init__()
        self.cs = channel_size
        self.cpm1 = nn.Conv2d( self.cs, 256, kernel_size=3, dilation=1, stride=1, padding=1)
        self.cpm2 = nn.Conv2d( self.cs, 256, kernel_size=3, dilation=2, stride=1, padding=2)
        self.cpm3 = nn.Conv2d( 256, 128, kernel_size=3, dilation=1, stride=1, padding=1)
        self.cpm4 = nn.Conv2d( 256, 128, kernel_size=3, dilation=2, stride=1, padding=2)
        self.cpm5 = nn.Conv2d( 128, 128, kernel_size=3, dilation=1, stride=1, padding=1)

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.relu4 = nn.ReLU(inplace=True)
        self.relu5 = nn.ReLU(inplace=True)
    def forward(self, x):
        x1_1 = self.relu1(self.cpm1(x))
        x1_2 = self.relu2(self.cpm2(x))
        x2_1 = self.relu3(self.cpm3(x1_2))
        x2_2 = self.relu4(self.cpm4(x1_2))
        x3_1 = self.relu5(self.cpm5(x2_2))
        return flops_counter.cat([x1_1, x2_1, x3_1] , 0)

class DSFD(nn.Module):
    def __init__(self):
        super(DSFD, self).__init__()
        self.size = 640
        self.num_classes = 2

        ######
        # build backbone
        ######
        resnet152 = vision.models.resnet152()
        self.layer1 = nn.Sequential(resnet152.conv1, resnet152.bn1, resnet152.relu, resnet152.maxpool, resnet152.layer1)
        self.layer2 = nn.Sequential(resnet152.layer2)
        self.layer3 = nn.Sequential(resnet152.layer3)
        self.layer4 = nn.Sequential(resnet152.layer4)
        self.layer5 = nn.Sequential(                                      
               *[nn.Conv2d(2048, 512, kernel_size=1),                         
                  nn.BatchNorm2d(512),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(512,512, kernel_size=3,padding=1,stride=2),
                  nn.BatchNorm2d(512),
                  nn.ReLU(inplace=True)]
        )
        self.layer6 = nn.Sequential(
               *[nn.Conv2d(512, 128, kernel_size=1,),
                  nn.BatchNorm2d(128),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(128, 256, kernel_size=3,padding=1,stride=2),
                  nn.BatchNorm2d(256),
                  nn.ReLU(inplace=True)]
        )

        ######
        # dsfd specific layers
        ######
        output_channels = [256, 512, 1024, 2048, 512, 256]
        # fpn
        fpn_in = output_channels

        self.latlayer3 = nn.Conv2d( fpn_in[3], fpn_in[2], kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( fpn_in[2], fpn_in[1], kernel_size=1, stride=1, padding=0)
        self.latlayer1 = nn.Conv2d( fpn_in[1], fpn_in[0], kernel_size=1, stride=1, padding=0)

        self.smooth3 = nn.Conv2d( fpn_in[2], fpn_in[2], kernel_size=1, stride=1, padding=0)
        self.smooth2 = nn.Conv2d( fpn_in[1], fpn_in[1], kernel_size=1, stride=1, padding=0)
        self.smooth1 = nn.Conv2d( fpn_in[0], fpn_in[0], kernel_size=1, stride=1, padding=0)

        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.eltmul = nn.EltMul()

        # fem
        cpm_in = output_channels
        self.cpm3_3 = FEM(cpm_in[0])
        self.cpm4_3 = FEM(cpm_in[1])
        self.cpm5_3 = FEM(cpm_in[2])
        self.cpm7 = FEM(cpm_in[3])
        self.cpm6_2 = FEM(cpm_in[4])
        self.cpm7_2 = FEM(cpm_in[5])

        # pa
        cfg_mbox = [1, 1, 1, 1, 1, 1]
        head = pa_multibox(output_channels, cfg_mbox, self.num_classes)
        
        # detection head
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        loc = list()
        conf = list()

        ######
        # Backbone
        ######
        conv3_3_x = self.layer1(x)
        conv4_3_x = self.layer2(conv3_3_x)
        conv5_3_x = self.layer3(conv4_3_x)
        fc7_x = self.layer4(conv5_3_x)
        conv6_2_x = self.layer5(fc7_x)
        conv7_2_x = self.layer6(conv6_2_x)

        ######
        # dsfd specific layers
        ######
        # fpn
        lfpn3_fc7_x_up = self.upsample(self.latlayer3(fc7_x))
        lfpn3_conv5_3_x = self.smooth3(conv5_3_x)
        if lfpn3_fc7_x_up[1] != lfpn3_conv5_3_x[1] or lfpn3_fc7_x_up[2] != lfpn3_conv5_3_x[2]:
            pad = (0, abs(lfpn3_conv5_3_x[2] - lfpn3_fc7_x_up[2]), 0, abs(lfpn3_conv5_3_x[1] - lfpn3_fc7_x_up[1]))
            lfpn3_fc7_x_up = F.pad(lfpn3_fc7_x_up, pad)
        lfpn3 = self.eltmul(lfpn3_fc7_x_up, lfpn3_conv5_3_x)

        lfpn2_lfpn3_up = self.upsample(self.latlayer2(lfpn3))
        lfpn2_conv4_3_x = self.smooth2(conv4_3_x)
        if lfpn2_lfpn3_up[1] != lfpn2_conv4_3_x[1] or lfpn2_lfpn3_up[2] != lfpn2_conv4_3_x[2]:
            pad = (0, abs(lfpn2_conv4_3_x[2] - lfpn2_lfpn3_up[2]), 0, abs(lfpn2_conv4_3_x[1] - lfpn2_lfpn3_up[1]))
            lfpn2_lfpn3_up = F.pad(lfpn2_lfpn3_up, pad)
        lfpn2 = self.eltmul(lfpn2_lfpn3_up, lfpn2_conv4_3_x)

        lfpn1_lfpn2_up = self.upsample(self.latlayer1(lfpn2))
        lfpn1_conv3_3_x = self.smooth1(conv3_3_x)
        if lfpn1_lfpn2_up[1] != lfpn1_conv3_3_x[1] or lfpn1_lfpn2_up[2] != lfpn1_conv3_3_x[2]:
            pad = (0, abs(lfpn1_conv3_3_x[2] - lfpn1_lfpn2_up[2]), 0, abs(lfpn1_conv3_3_x[1] - lfpn1_lfpn2_up[1]))
            lfpn1_lfpn2_up = F.pad(lfpn1_lfpn2_up, pad)
        lfpn1 = self.eltmul(lfpn1_lfpn2_up, lfpn1_conv3_3_x)

        conv5_3_x = lfpn3
        conv4_3_x = lfpn2
        conv3_3_x = lfpn1

        # fem
        sources = [conv3_3_x, conv4_3_x, conv5_3_x, fc7_x, conv6_2_x, conv7_2_x]
        sources[0] = self.cpm3_3(sources[0])
        sources[1] = self.cpm4_3(sources[1])
        sources[2] = self.cpm5_3(sources[2])
        sources[3] = self.cpm7(sources[3])
        sources[4] = self.cpm6_2(sources[4])
        sources[5] = self.cpm7_2(sources[5])

        # apply multibox head to source layers
        conf = list()
        for x, l, c in zip(sources, self.loc, self.conf):
            l(x)
            # mio: max_in_out
            conf.append(c(x))
        face_conf = flops_counter.cat([flops_counter.view([o[1], o[2], 2], (1, -1)) for o in conf], 1)
        output = self.softmax(flops_counter.view(face_conf, (1, -1, 2)))
        return output


def pa_multibox(output_channels, mbox_cfg, num_classes):
    loc_layers = []
    conf_layers = []
    for k, v in enumerate(output_channels):
        input_channels = 512
        if k ==0:
            loc_output = 4
            conf_output = 2
        elif k==1:
            loc_output = 8
            conf_output = 4
        else:
            loc_output = 12
            conf_output = 6
        loc_layers += [DeepHeadModule(input_channels, mbox_cfg[k] * loc_output)]
        conf_layers += [DeepHeadModule(input_channels, mbox_cfg[k] * (2+conf_output))]
    return (loc_layers, conf_layers)


class DeepHeadModule(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(DeepHeadModule , self).__init__()
        self._input_channels = input_channels
        self._output_channels = output_channels
        self._mid_channels = min(self._input_channels, 256)
        #print(self._mid_channels)
        self.conv1 = nn.Conv2d( self._input_channels, self._mid_channels, kernel_size=3, dilation=1, stride=1, padding=1)
        self.conv2 = nn.Conv2d( self._mid_channels, self._mid_channels, kernel_size=3, dilation=1, stride=1, padding=1)
        self.conv3 = nn.Conv2d( self._mid_channels, self._mid_channels, kernel_size=3, dilation=1, stride=1, padding=1)
        self.conv4 = nn.Conv2d( self._mid_channels, self._output_channels, kernel_size=1, dilation=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        # return self.conv4(F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1(x), inplace=True)), inplace=True)), inplace=True))
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.conv4(out)
        return out