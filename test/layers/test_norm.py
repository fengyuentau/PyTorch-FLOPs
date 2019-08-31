from _utils import test_on

import sys
sys.path.append('.')
from flops_counter import nn
from flops_counter.tensorsize import TensorSize

######
# test on BatchNorm2d
######
bn2d = {
    'layers': [
        nn.BatchNorm2d(64) # same shape
    ],
    'ins': [
        TensorSize([1, 64, 112, 112])
    ],
    'out_shape': [
        TensorSize([1, 64, 112, 112])
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
        TensorSize([1, 256, 56, 56])
    ],
    'out_shape': [
        TensorSize([1, 256, 56, 56])
    ],
    'out_flops': [
        2408448
    ]
}

test_on(l2norm2d)