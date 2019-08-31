from _utils import test_on

import sys
sys.path.append('.')
from flops_counter import nn
from flops_counter.tensorsize import TensorSize

######
# test on Upsample
######
upsample = {
    'layers': [
        nn.Upsample(scale_factor=2, mode='bilinear') # same shape
    ],
    'ins': [
        TensorSize([1, 1024, 20, 20])
    ],
    'out_shape': [
        TensorSize([1, 1024, 40, 40])
    ],
    'out_flops': [
        15974400
    ]
}

test_on(upsample)