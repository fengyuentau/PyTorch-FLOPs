import sys
sys.path.append('.')
import flops_counter as fc
from models import LFFDv1, LFFDv2

lffdv1 = LFFDv1()
x = fc.TensorSize([1, 3, 224, 224])
y = lffdv1(x)
print(y, lffdv1.flops)
# print(dsfd)


lffdv2 = LFFDv2()
x = fc.TensorSize([1, 3, 224, 224])
y = lffdv2(x)
print(y, lffdv2.flops)
# print(dsfd)