import sys
sys.path.append('.')
import flops_counter as fc
from models import FaceBoxes

model = FaceBoxes()
x = fc.TensorSize([1, 3, 224, 224])
y = model(x)
print(y, model.flops)
# print(dsfd)