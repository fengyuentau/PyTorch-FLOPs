from functools import reduce

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

# TODO: create a basic data type and move permute into the definition of that data type
def permute(input, desired_shape: tuple):
    assert len(input) == len(desired_shape), \
        'Dimension of input {:d} does not match the one {:d} of desired_shape .'.format(len(input), len(desired_shape))

    output = [x for x in input]
    for v in desired_shape:
        idx = desired_shape.index(v)
        output[idx] = input[v]

    return output

# TODO: create a basic data type and move view into the definition of that data type
def view(input, shape):
    if len(shape) == 2 and shape[0] == 1 and shape[1] == -1:
        return [1] + [reduce((lambda a, b: a * b), input)]
    elif len(shape) == 3 and shape[0] == 1 and shape[1] == -1 and shape[2] == 2:
        assert reduce((lambda a, b: a * b), input) % shape[2] == 0, 'The multiple output of input shape must be divisible by the given last shape dim.'
        return [1] + [reduce((lambda a, b: a * b), input) // shape[2]] + [2]