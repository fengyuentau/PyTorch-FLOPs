import sys
sys.path.append('.')
import flops_counter as fc
from vision.models import MobileNetV1

model = MobileNetV1()
x = fc.TensorSize([1, 3, 224, 210])
y = model(x)
print(y, model.flops)
print(model)