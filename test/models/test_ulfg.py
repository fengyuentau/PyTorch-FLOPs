import sys
sys.path.append('.')
import flops_counter as fc
from models import ULFG

model = ULFG()
x = fc.TensorSize([1, 3, 224, 224])
y = model(x)
# print(y, model.flops)
print(model.flops)
# print(model.name)
# print(model)
# print(model.backbone.flops)

model = ULFG('rfb')
x = fc.TensorSize([1, 3, 224, 224])
y = model(x)
# print(y, model.flops)
# # print(model.name)
print(model.flops)