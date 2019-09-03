import sys
sys.path.append('.')

import flops_counter
x = flops_counter.TensorSize(1,3,640,640)

from models.pyramidbox import PyramidBox
pb = PyramidBox(300)

y = pb(x)

print(y, pb.flops)
print(pb)
