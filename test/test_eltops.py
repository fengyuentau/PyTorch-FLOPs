from _utils import test_on

import sys
sys.path.append('.')
from flops_counter import nn

######
# test on EltAdd
######
eltadd = {
    'layers': [
        nn.EltAdd() # same shape
    ],
    'ins': [
        [64, 112, 112]
    ],
    'out_shape': [
        [64, 112, 112]
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
        [64, 112, 112]
    ],
    'out_shape': [
        [64, 112, 112]
    ],
    'out_flops': [
        802816
    ]
}

test_on(eltmul)