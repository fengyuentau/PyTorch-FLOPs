import sys
sys.path.append('.')
import flops_counter as fc
from models.dsfd import DSFD

model = DSFD()
x = fc.TensorSize([1, 3, 224, 224])
y = model(x)
print(y, model.flops)
# print(dsfd)