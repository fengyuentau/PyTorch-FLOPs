from flops_counter.tensorsize import TensorSize

def cat(tensor_sizes, dim: int):
    assert isinstance(tensor_sizes, list) or isinstance(tensor_sizes, tuple), 'tensors must be either list or tuple.'

    _dim = tensor_sizes[0].dim
    for t in tensor_sizes:
        assert _dim == t.dim, 'TensorSize(s) must have the same dimension.'

    for i in range(_dim):
        if i != dim:
            _s = tensor_sizes[0].value[i]
            for t in tensor_sizes:
                assert _s == t.value[i], 'tensors must be of the same shape.'

    assert dim in [i for i in range(_dim)], 'Given dim {:d} is out of shape of TensorSize(s) {:d}.'.format(dim, _dim)

    y = tensor_sizes[0].value
    for i in range(1, len(tensor_sizes)):
        y[dim] += tensor_sizes[i].value[dim]

    return TensorSize(y)