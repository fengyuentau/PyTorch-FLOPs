import sys
sys.path.append('.')
import flops_counter as fc
from models import LFFD

lffdv1 = LFFD('v1')
x = fc.TensorSize([1, 3, 224, 210])
y = lffdv1(x)
print(y, lffdv1.flops)
# print(dsfd)


lffdv2 = LFFD('v2')
x = fc.TensorSize([1, 3, 224, 210])
y = lffdv2(x)
print(y, lffdv2.flops)
# print(dsfd)