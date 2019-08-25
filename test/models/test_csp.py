import sys
sys.path.append('.')
from models.csp import CSP

csp = CSP()

x = [3, 64, 64]
y = csp(x)
print(csp.flops)

x = [3, 224, 224]
y = csp(x)
print(csp.flops)



# print(y)
# print(csp)
