def pad(input, pad, mode='constant', value=0):
    assert len(pad) % 2 == 0, 'Padding length must be divisible by 2'
    assert len(pad) // 2 <= len(input), 'Padding length too large'

    if len(pad) == 4 and len(input) == 3:
        cin, hin, win = input
        pleft, pright, ptop, pbottom = pad

        cout = cin
        hout = hin + ptop + pbottom
        wout = win + pleft + pright
        y = [cout, hout, wout]
        return y
    else:
        raise NotImplementedError