import sys
sys.path.append('.')
import flops_counter as fc
from models import EXTD

model = EXTD()
x = fc.TensorSize([1, 3, 224, 210])
y = model(x)
print(y, model.flops)