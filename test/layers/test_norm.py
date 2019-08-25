from _utils import test_on

import sys
sys.path.append('.')
from flops_counter import nn

######
# test on BatchNorm2d
######
bn2d = {
    'layers': [
        nn.BatchNorm2d(64) # same shape
    ],
    'ins': [
        [64, 112, 112]
    ],
    'out_shape': [
        [64, 112, 112]
    ],
    'out_flops': [
        4816896
    ]
}

test_on(bn2d)

######
# test on L2Norm2d
######
l2norm2d = {
    'layers': [
        nn.L2Norm2d(256) # same shape
    ],
    'ins': [
        [256, 56, 56]
    ],
    'out_shape': [
        [256, 56, 56]
    ],
    'out_flops': [
        2408448
    ]
}

test_on(l2norm2d)