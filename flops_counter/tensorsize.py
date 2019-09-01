import copy
from functools import reduce

from flops_counter._six import container_abcs

# refer to torch/tensor.py in pytorch@3ed720079
# see also here: https://github.com/pytorch/pytorch/issues/2013
#  and https://stackoverflow.com/questions/49724954/how-are-pytorchs-tensors-implemented

def _infer_sizes(sizes, total):
    to_infer = -1
    total_sizes = 1
    for i, size in enumerate(sizes):
        total_sizes *= size
        if size == -1:
            if to_infer >= 0:
                raise RuntimeError
            to_infer = i
    if to_infer >= 0:
        assert total % total_sizes == 0, "Can't make sizes have exactly %d elements" % total
        sizes = list(sizes)
        sizes[to_infer] = -total // total_sizes
    return sizes

class TensorSize(object):
    def __init__(self, *sizes):
        assert 0 not in sizes, \
            'Non of the dim should be 0.'

        # dim > 0, dim -> int
        self._tensor_size = list()
        for i, v in enumerate(sizes):
            if isinstance(v, container_abcs.Iterable):
                if not isinstance(v[0], container_abcs.Iterable):
                    self._tensor_size = [int(i) for i in v]
                else:
                    raise TypeError('flops_counter.TensorSize() takes an iterable of \'int\' (item {:d} is \'{:s}\')'.format(i, type(v[i]).__name__))
            else:
                self._tensor_size.append(int(v))
            

    def __repr__(self):
        return 'flops_counter.TensorSize({:s})'.format(str(self._tensor_size))

    def view(self, *args):
        if len(args) == 1 and isinstance(args[0], list):
            dst_ = args[0]
        else:
            dst_ = list(args)

        dst = TensorSize(_infer_sizes(dst_, self.nelement))
        assert dst.nelement == self.nelement, \
            'Elements must be equal.'

        return dst

    def view_as(self, tensor_size):
        return self.view(tensor_size.value)

    def permute(self, *dims):
        perm = list(dims)
        assert len(perm) == len(self._tensor_size), 'Invalid permutation'

        _output = copy.deepcopy(self._tensor_size)
        for i, v in enumerate(perm):
            _output[i] = self._tensor_size[v]

        return TensorSize(_output)

    @property
    def nelement(self):
        return reduce((lambda a, b: a * b), self._tensor_size)

    @property
    def value(self):
        return self._tensor_size

    @property
    def dim(self):
        return len(self._tensor_size)