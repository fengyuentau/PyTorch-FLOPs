import sys
sys.path.append('.')
from flops_counter import nn
from flops_counter.tensorsize import TensorSize

mlist = nn.ModuleList([
    nn.Conv2d(3, 64, 3, 1, 1)
])

ts = TensorSize([1, 3, 224, 224])

# print(mlist[0](ts))
# print(mlist[0].flops)
# print(mlist.flops)


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()

        self.conv = nn.Conv2d(3, 64, 3, 1, 1)
        self.headers = nn.ModuleList([
            nn.Conv2d(64, 4, 3, 1, 1),
            nn.Conv2d(64, 2, 3, 1, 1)
        ])

    def forward(self, x):
        x = self.conv(x)
        y1 = self.headers[0](x)
        y2 = self.headers[1](x)
        # self.headers.settle(x, y1)
        return y1, y2


model = NN()
o = model(ts)
print(model)
# print(model.flops)
# print(model.conv.flops)
# print(model.headers.flops)
# print(model.headers[0].flops)
# print(model.headers[1].flops)