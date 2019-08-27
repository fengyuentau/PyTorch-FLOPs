import itertools
import sys
import builtins


PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3
PY37 = sys.version_info[0] == 3 and sys.version_info[1] == 7

if PY2:
    string_classes = basestring
else:
    string_classes = (str, bytes)

if PY2:
    import collections
    container_abcs = collections
elif PY3:
    import collections.abc
    container_abcs = collections.abc