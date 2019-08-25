def test_on(package):
    for input, layer, gt_out, gt_flops in zip(package['ins'], package['layers'], package['out_shape'], package['out_flops']):
        output = layer(input)

        assert output == gt_out, \
            '[x] Test FAILED at {:s} with input {:s}. Expected output {:s} and flops {:d}, but output {:s} and flops {:d} given.'.format(
                str(layer), str(input), str(gt_out), gt_flops, str(output), tconv.flops)
    print('[o] Test PASSED on {:s}.'.format(repr(package)))