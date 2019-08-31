from .nn import *
from .utils import cat, permute, view

import sys

PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3
PY37 = sys.version_info[0] == 3 and sys.version_info[1] == 7

if PY2:
    string_classes = basestring
else:
    string_classes = (str, bytes)

def typename(o):
    if isinstance(o, list):
        return type(o)

    module = ''
    class_name = ''
    if hasattr(o, '__module__') and o.__module__ != 'builtins' \
            and o.__module__ != '__builtin__' and o.__module__ is not None:
        module = o.__module__ + '.'

    if hasattr(o, '__qualname__'):
        class_name = o.__qualname__
    elif hasattr(o, '__name__'):
        class_name = o.__name__
    else:
        class_name = o.__class__.__name__

    return module + class_name