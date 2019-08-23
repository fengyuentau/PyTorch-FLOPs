import sys

PY_VER = sys.version_info[0]
if PY_VER == 2:
    import collections
    container_abcs = collections
elif PY_VER == 3:
    import collections.abc
    container_abcs = collections.abc

from itertools import repeat


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            x_cast = tuple()
            for _x in x:
                if not isinstance(_x, int):
                    _x = dtype(_x)
                x_cast += (_x,)
            return x_cast
        if isinstance(x, int):
            x = int(x)
            return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)
