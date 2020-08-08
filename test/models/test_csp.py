import sys
sys.path.append('.')
import flops_counter as fc
from models import CSP

model = CSP()
# input size either height or width must be dividable by 16,
# this is due to the model design
x = fc.TensorSize([1, 3, 224, 112])
y = model(x)
print(y, model.flops)
# print(model)