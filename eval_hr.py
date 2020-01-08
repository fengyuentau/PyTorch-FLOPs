import os
import os.path as osp
import argparse
import sys
import time

import cv2
import numpy as np

import flops_counter

import platform
ossystem = platform.system()
osrelease = platform.release()
DEFAULT_WIDERFACE_ROOT = '/Users/fengyuantao/playground/dataset/WIDER_FACE'
if ossystem == 'Darwin':
    DEFAULT_WIDERFACE_ROOT = '/Users/fengyuantao/playground/dataset/WIDER_FACE'
elif ossystem == 'Linux':
    if osrelease == '4.15.0-54-generic':
        DEFAULT_WIDERFACE_ROOT = '/home/tau/Documents/dataset/WiderFace'
    elif osrelease == '4.4.0-87-generic':
        DEFAULT_WIDERFACE_ROOT = '/home1/tau/datasets/wider_face'
elif ossystem == 'Windows':
    DEFAULT_WIDERFACE_ROOT = 'D:\dataset\widerface'

def parseargs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--widerface_root', default=DEFAULT_WIDERFACE_ROOT)
    parser.add_argument('--set', default='val', help='Run in mode test or val.')

    args = parser.parse_args()
    return args

def get_set_size(annotation_filepath):
    cnt = 0
    with open(annotation_filepath, 'r') as annofile:
        for line in annofile:
            if '.jpg' in line:
                cnt += 1
    return cnt

def build_net():
    from models import HR
    net = HR()
    return net


def calc_flops(net, img, scale=1.0, flip=True, max_downsample=16):
    img_s = img
    if scale != 1.0:
        img_s = cv2.resize(img, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    img_s_h, img_s_w, img_s_c = img_s.shape
    # skip when input resolution is too huge, see line 268 in cnn_widerface_test_AB.m
    if scale > 1 and (img_s_h > 5000 or img_s_w > 5000):
        return 0

    x = flops_counter.TensorSize([1, img_s_c, img_s_h, img_s_w])

    net(x)
    flops = net.flops * 2 if flip else net.flops

    net.set_flops_zero()
    return flops

if __name__ == '__main__':
    args = parseargs()

    WIDERFACE_ROOT = args.widerface_root
    WIDERFACE_VAL = {
        'IMGROOT': osp.join(WIDERFACE_ROOT, 'WIDER_val', 'images'),
        'ANNOTATION': osp.join(WIDERFACE_ROOT, 'wider_face_split', 'wider_face_val_bbx_gt.txt')
    }
    WIDERFACE_TEST = {
        'IMGROOT': osp.join(WIDERFACE_ROOT, 'WIDER_test', 'images'),
        'ANNOTATION': osp.join(WIDERFACE_ROOT, 'wider_face_split', 'wider_face_test_filelist.txt')
    }

    assert isinstance(args.set, str), 'Set must be included and must be string.'
    widerface_set = args.set.lower()
    assert widerface_set == 'val' or widerface_set == 'test', 'Mode must be either val or test.'
    print('-> Running on WIDER Face {:s} set.'.format(widerface_set))
    IMGROOT = WIDERFACE_VAL['IMGROOT'] if widerface_set == 'val' else WIDERFACE_TEST['IMGROOT']
    ANNOTATION = WIDERFACE_VAL['ANNOTATION'] if widerface_set == 'val' else WIDERFACE_TEST['ANNOTATION']


    net = build_net()

    scales = [-2, -1, 0, 1]
    # scales = [0]

    img_cnt = 0
    img_total = get_set_size(ANNOTATION)
    flops_avg = 0
    flops_total = int(0)
    with open(ANNOTATION, 'r') as annofile:
        for line in annofile:
            if '.jpg' in line:
                # read the image
                imgpath = osp.join(IMGROOT, line.strip()) # the name of the first image: /0--Parade/0_Parade_marchingband_1_737.jpg
                img = cv2.imread(imgpath) # img.shape <- [height, width, channel]

                flops = int(0)
                # run profile at set ratio
                for s in scales:
                    flops += calc_flops(net, img, 2.0**s, flip=False)

                # statistics
                img_cnt += 1
                flops_total += flops
                flops_avg = int(flops_total / img_cnt)
                # simple progress bar
                sys.stdout.write('\r')
                sys.stdout.write('-> Profiling model {:s}:: {:d}/{:d}, avg FLOPs: {:,d}, total FLOPs: {:,d}'.format(net.name, img_cnt, img_total, flops_avg, flops_total))
                sys.stdout.flush()
    print('\n')