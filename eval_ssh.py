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
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()

    parser.add_argument('--widerface_root', default=DEFAULT_WIDERFACE_ROOT)
    parser.add_argument('--set', default='val', help='Run in mode test or val.')
    parser.add_argument('--pyramid', default=False, type=str2bool)

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
    from models import SSH
    net = SSH()
    return net

def calc_scale(image_shape, target_size, max_size):
    im_size_min = min(image_shape)
    im_size_max = max(image_shape)

    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if int(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    return im_scale

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

    scales = [1200, ]
    max_size = 1600
    if args.pyramid:
        scales = [500, 800, 1200, 1600, ]
        max_size = -1
        pyramid_base_size = [800,1200]


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

                pyramid or not
                if args.pyramid:
                    base_scale = calc_scale(img.shape[0:2], pyramid_base_size[0], pyramid_base_size[1])
                    pyramid_scales = [ float(scale)/pyramid_base_size[0]*base_scale for scale in scales]
                    for pyramid_scale in pyramid_scales:
                        flops += calc_flops(net, img, pyramid_scale)
                else:
                    im_scale = calc_scale(img.shape[0:2], scales[0], max_size)
                    flops += calc_flops(net, img, im_scale)
                # flops += calc_flops(net, img, 1)

                # statistics
                img_cnt += 1
                flops_total += flops
                flops_avg = int(flops_total / img_cnt)
                # simple progress bar
                sys.stdout.write('\r')
                sys.stdout.write('-> Profiling model {:s} with pyramid={} : {:d}/{:d}, avg FLOPs: {:,d}, total FLOPs: {:,d}'.format(net.name, str(args.pyramid), img_cnt, img_total, flops_avg, flops_total))
                sys.stdout.flush()
    print('\n')