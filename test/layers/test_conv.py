from _utils import test_on

import sys
sys.path.append('.')
from flops_counter import nn
from flops_counter.tensorsize import TensorSize


######
# test on Conv2d
######
conv = {
    'layers': [
        nn.Conv2d(3, 64, 7, 2, 3, bias=False), # half of the shape
        nn.Conv2d(64, 64, 1, 1, 0, bias=False), # same shape
        nn.Conv2d(64, 64, 3, 1, 1, bias=False) # same shape
    ],
    'ins': [
        TensorSize([1, 3, 224, 224]),
        TensorSize([1, 64, 56, 56]),
        TensorSize([1, 64, 56, 56])
    ],
    'out_shape': [
        TensorSize([1, 64, 112, 112]),
        TensorSize([1, 64, 56, 56]),
        TensorSize([1, 64, 56, 56])
    ],
    'out_flops': [
        235225088,
        25489408,
        231010304
    ]
}

test_on(conv)


######
# test on ConvTranspose2d
######
convtran = {
    'layers': [
        nn.ConvTranspose2d(512, 256, 4, 2, 1), # double the shape, except channels
        nn.ConvTranspose2d(1024, 256, 4, 4, 0) # quadro the shape, except channels
    ],
    'ins': [
        TensorSize([1, 512, 28, 28]),
        TensorSize([1, 1024, 14, 14])
    ],
    'out_shape': [
        TensorSize([1, 256, 56, 56]),
        TensorSize([1, 256, 56, 56])
    ],
    'out_flops': [
        13153337344,
        26306674688
    ]
}

test_on(convtran)