import sys
sys.path.append('..')

import flops_counter
import flops_counter.nn as nn

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
        x3_1 = self.relu5(self.cmp5(x2_2))
        return flops_counter.cat([x1_1, x2_1, x3_1] , 0)

class DSFD(nn.Module):
    def __init__(self):
        super(DSFD, self).__init__()
        self.size = 640
        self.num_classes = 2