import sys
sys.path.append('..')

from models.csp import CSP

csp_resnet50 = CSP()

input = [3, 224, 224]
flops = csp_resnet50(input)

print(flops)