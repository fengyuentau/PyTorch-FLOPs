import os
import os.path as osp
import argparse
import sys
import time

import numpy as np
import cv2
import math

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
    from models import SRN
    net = SRN()
    return net


def calc_flops(net, image, shrink=1, flip=False, max_downsample=16):
    image_shape = image.shape
    if shrink != 1:
        x = cv2.resize(img, None, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_LINEAR)
        image_shape = x.shape

    h, w, c = image_shape
    x = flops_counter.TensorSize([1, c, h, w])

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
    largest_input = 2100 * 2100

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

                flops_det01 = int(0)
                flops_det2 = int(0)
                flops_det3 = int(0)
                flops_det4 = int(0)
                img_h, img_w, img_c = img.shape
                if img_h * img_w < largest_input:
                    # det0 and det1 (0-original, 1-flip)
                    flops_det01 = calc_flops(net, img, 1, flip=False)
                    # det2 (shrink 0.5)
                    flops_det2 = calc_flops(net, img, 0.5, flip=False)
                    # det3 (enlarge)
                    enlarge_time = int(math.floor(math.log(largest_input / img_w / img_h, 2.25)))
                    for t in range(enlarge_time):
                        resize_scale = math.pow(1.5, t+1)
                        flops_det3 += calc_flops(net, img, resize_scale, flip=False)
                    # det4 (final ratio)
                    final_ratio = math.sqrt(largest_input / img_h / img_w)
                    flops_det4 = calc_flops(net, img, final_ratio, flip=False)
                else:
                    largest_ratio = math.sqrt(largest_input / img_w / img_h)
                    # det0 and det1 (0-largest, 1-largest's flip)
                    flops_det01 = calc_flops(net, img, largest_ratio, flip=False)
                    # det2 (shrink 0.75)
                    flops_det2 = calc_flops(net, img, 0.75, flip=False)
                    # det3 (shrink 0.5)
                    flops_det3 = calc_flops(net, img, 0.5, flip=False)
                    # det4 (no det4)

                # sum
                flops = flops_det01 + flops_det2 + flops_det3 + flops_det4

                # statistics
                img_cnt += 1
                flops_total += flops
                flops_avg = int(flops_total / img_cnt)
                # simple progress bar
                sys.stdout.write('\r')
                sys.stdout.write('-> Profiling model {:s}:: {:d}/{:d}, avg FLOPs: {:,d}, total FLOPs: {:,d}'.format(net.name, img_cnt, img_total, flops_avg, flops_total))
                sys.stdout.flush()
    print('\n')