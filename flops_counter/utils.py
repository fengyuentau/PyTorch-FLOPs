def cat(tensors, dim: int):
    assert isinstance(tensors, list) or isinstance(tensors, tuple), 'tensors must be either list or tuple.'

    _size = len(tensors[0])
    for t in tensors:
        assert _size == len(t), 'tensors must have the same dimension.'

    for i in range(_size):
        if i != dim:
            d = tensors[0][i]
            for t in tensors:
                assert d == t[i], 'tensors must be of the same shape.'

    assert dim in [i for i in range(len(tensors))], 'dim is out of tensor\'s shape.'

    y = [i for i in tensors[0]]
    for i in range(1, len(tensors)):
        y[dim] += tensors[i][dim]

    return y