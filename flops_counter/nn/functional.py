from flops_counter.tensorsize import TensorSize

def pad(input, pad, mode='constant', value=0):
    assert isinstance(input, TensorSize), \
            'Type of input must be \'{}\'.'.format(TensorSize.__name__)
    assert len(pad) % 2 == 0, 'Padding length must be divisible by 2'
    assert len(pad) // 2 <= input.dim, 'Padding length too large'

    if len(pad) == 4 and input.dim == 4:
        bsin, cin, hin, win = input.value
        pleft, pright, ptop, pbottom = pad

        bsout = bsin
        cout = cin
        hout = hin + ptop + pbottom
        wout = win + pleft + pright
        y = TensorSize([bsout, cout, hout, wout])
        return y
    else:
        raise NotImplementedError