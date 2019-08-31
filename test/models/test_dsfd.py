import sys
sys.path.append('.')
from models.dsfd import DSFD

dsfd = DSFD()

# x = [3, 64, 64]
# y = dsfd(x)
# print(dsfd.flops)

x = [3, 224, 224]
y = dsfd(x)
print(y, dsfd.flops)
print(dsfd)