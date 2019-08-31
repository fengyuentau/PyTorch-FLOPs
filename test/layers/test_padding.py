import sys
sys.path.append('.')
from flops_counter import nn
from flops_counter.tensorsize import TensorSize


padding1 = nn.ZeroPad2d((2,0,0,2))
padding2 = nn.ZeroPad2d((0,2,2,0))

x = TensorSize([1, 3, 224, 224])
expected_y = TensorSize([1, 3, 226, 226])

y1 = padding1(x)
y2 = padding2(x)

print(padding1, y1)
print(padding2, y2)

assert y1.value == expected_y.value, 'padding1 wrong'
assert y2.value == expected_y.value, 'padding2 wrong'