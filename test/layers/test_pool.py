from _utils import test_on

import sys
sys.path.append('.')
from flops_counter import nn

######
# test on MaxPool2d
######
mxpool2d = {
    'layers': [
        nn.MaxPool2d(3, 2, 1) # same shape
    ],
    'ins': [
        [64, 112, 112]
    ],
    'out_shape': [
        [64, 56, 56]
    ],
    'out_flops': [
        602112
    ]
}

test_on(mxpool2d)