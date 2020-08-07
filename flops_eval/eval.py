import os
import sys
import math
import argparse

import numpy as np

import importlib

sys.path.append('.')
import flops_counter
import flops_counter.nn as nn

import models

from data import WIDERFace


def parseargs():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser('FLOPs Evaluation for Models on WIDER Face.')

    parser.add_argument('--widerface_root',
                        default='data/widerface',
                        help='Root of WIDER Face dataset.')
    parser.add_argument('--split',
                        default='val',
                        choices=['val-easy', 'val-medium', 'val-hard', 'val', 'test'],
                        help='Run FLOPs evaluation on WIDER Face validation set or test set.')
    parser.add_argument('--model',
                        default='CSP',
                        help='Evaluate the specified model.')
    parser.add_argument('--multi_scale_testing',
                        default=True,
                        type=str2bool,
                        help='Evaluate in multi scales or a single scale 1.')

    args = parser.parse_args()
    return args

def to_scientific(flops):
    p = math.floor(math.log(flops, 2))
    rank = p // 10
    left = p % 10
    dim = ''
    if rank == 1:
        dim = 'K'
    elif rank == 2:
        dim = 'M'
    elif rank == 3:
        dim = 'G'
    elif rank == 4:
        dim = 'P'
    v = flops
    for i in range(rank * 10):
        v = v / 2
    return '{} {}'.format(str(v), dim)

def main(args):
    # build dataset
    dataset = WIDERFace(widerface_root=args.widerface_root, split=args.split)

    # build model
    model  = eval('models.'+args.model)()

    # flops_eval
    mod = importlib.import_module('core.'+args.model.lower())
    avg_flops = mod.flops_eval(dataset, model)
    print(avg_flops)
    print('Average FLOPs: {}FLOPs'.format(to_scientific(avg_flops)))




if __name__ == '__main__':
    args = parseargs()
    main(args)