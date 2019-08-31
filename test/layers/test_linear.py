from _utils import test_on

import sys
sys.path.append('.')
from flops_counter import nn
from flops_counter.tensorsize import TensorSize

######
# test on Linear
######
linear = {
    'layers': [
        nn.Linear(4096, 8192) # same shape
    ],
    'ins': [
        TensorSize([1, 4096])
    ],
    'out_shape': [
        TensorSize([1, 8192])
    ],
    'out_flops': [
        67108864
    ]
}

test_on(linear)