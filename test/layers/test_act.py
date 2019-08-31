from _utils import test_on

import sys
sys.path.append('.')
from flops_counter import nn
from flops_counter.tensorsize import TensorSize

######
# test on ReLU
######
relu = {
    'layers': [
        nn.ReLU() # same shape
    ],
    'ins': [
        TensorSize([1, 64, 112, 112])
    ],
    'out_shape': [
        TensorSize([1, 64, 112, 112])
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
        TensorSize([1, 1, 56, 56])
    ],
    'out_shape': [
        TensorSize([1, 1, 56, 56])
    ],
    'out_flops': [
        9408
    ]
}

test_on(sigmoid)

######
# test on Softmax
######
softmax = {
    'layers': [
        nn.Softmax(dim=-1) # same shape
    ],
    'ins': [
        TensorSize([1, 4185, 2])
    ],
    'out_shape': [
        TensorSize([1, 4185, 2])
    ],
    'out_flops': [
        25108
    ]
}

test_on(softmax)