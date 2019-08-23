# Things need to be test:
# 1. whether the output size is correct
# 2. whether the output flops is correct

import sys
sys.path.append('..')

from flops_counter import nn

test_convs = [
    nn.Conv2d(3, 128, 3, 1, 1), # remain the same size but with channels changed
    nn.Conv2d(3, 128, 3, 2, 1), # height/width becomes half of the original
    nn.Conv2d(3, 512, 1, 1, 0) # remain the same size but with channels changed
]
test_convtrans = [
    nn.ConvTranspose2d(3, 128, 4, 2, 1)
]

test_ins = [
    [3, 64, 64],
    [3, 14, 14],
    [3, 7, 7]
]

for tin in test_ins:
    for tconv in test_convs:
        output, flops = tconv(tin)
        print('{:s}, input size: {:s}, output size: {:s}, flops: {:d}'.format(str(tconv), str(tin), str(output), flops))
    for tconvtran in test_convtrans:
        output, flops = tconvtran(tin)
        print('{:s}, input size: {:s}, output size: {:s}, flops: {:d}'.format(str(tconvtran), str(tin), str(output), flops))
        
    print()