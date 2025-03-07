from _utils import test_on

import sys
sys.path.append('.')
from flops_counter import nn
from flops_counter.tensorsize import TensorSize

######
# test on MaxPool2d
######
mxpool2d = {
    'layers': [
        nn.MaxPool2d(3, 2, 1) # same shape
    ],
    'ins': [
        TensorSize([1, 64, 112, 112])
    ],
    'out_shape': [
        TensorSize([1, 64, 56, 56])
    ],
    'out_flops': [
        602112
    ]
}

test_on(mxpool2d)