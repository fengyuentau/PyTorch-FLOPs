from _utils import test_on

import sys
sys.path.append('.')
from flops_counter import nn

######
# test on ReLU
######
relu = {
    'layers': [
        nn.ReLU() # same shape
    ],
    'ins': [
        [64, 112, 112]
    ],
    'out_shape': [
        [64, 112, 112]
    ],
    'out_flops': [
        1605632
    ]
}

test_on(relu)

######
# test on Sigmoid
######
sigmoid = {
    'layers': [
        nn.Sigmoid() # same shape
    ],
    'ins': [
        [1, 56, 56]
    ],
    'out_shape': [
        [1, 56, 56]
    ],
    'out_flops': [
        9408
    ]
}

test_on(sigmoid)