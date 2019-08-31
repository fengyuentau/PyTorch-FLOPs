from _utils import test_on

import sys
sys.path.append('.')
from flops_counter import nn
from flops_counter.tensorsize import TensorSize

######
# test on EltAdd
######
eltadd = {
    'layers': [
        nn.EltAdd() # same shape
    ],
    'ins': [
        TensorSize([64, 112, 112])
    ],
    'out_shape': [
        TensorSize([64, 112, 112])
    ],
    'out_flops': [
        802816
    ]
}

test_on(eltadd)

######
# test on EltMul
######
eltmul = {
    'layers': [
        nn.EltMul() # same shape
    ],
    'ins': [
        TensorSize([1, 64, 112, 112])
    ],
    'out_shape': [
        TensorSize([1, 64, 112, 112])
    ],
    'out_flops': [
        802816
    ]
}

test_on(eltmul)