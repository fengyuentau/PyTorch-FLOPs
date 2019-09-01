import os
import os.path as osp
import argparse
import sys
import time

import cv2
import numpy as np

import flops_counter

def parseargs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--widerface_root', default='/Users/fengyuantao/playground/dataset/WIDER_FACE')
    parser.add_argument('--set', default='val', help='Run in mode test or val.', required=True)
    # parser.add_argument('--max_downsample', default=32, type=int)

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
    from models import DSFD
    net = DSFD()
    return net

def calc_shrink(h, w):
    max_im_shrink = (0x7fffffff / 200.0 / (h * w)) ** 0.5 # the max size of input image for caffe
    max_im_shrink = 3 if max_im_shrink > 3 else max_im_shrink
    shrink = max_im_shrink if max_im_shrink < 1 else 1
    return shrink, max_im_shrink


def calc_flops(net, img, shrink=1, flip=True, max_downsample=16):
    x = img
    if shrink != 1:
        x = cv2.resize(img, None, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_LINEAR)
    h, w, c = x.shape

    x = flops_counter.TensorSize([1, c, h, w])
    net(x)

    flops = net.flops * 2 if flip else net.flops
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

                # calculate shrink
                shrink, max_im_shrink = calc_shrink(img.shape[0], img.shape[1])

                # det0 and det1 (det1 is the flipped version of det0)
                flops_det01 = calc_flops(net, img, shrink, flip=True)
                # det2
                st = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink
                flops_det2 = calc_flops(net, img, st, flip=False)
                if max_im_shrink > 0.75:
                    flops_det2 += calc_flops(net, img, 0.75, flip=False)
                # det3
                bt = min(2, max_im_shrink) if max_im_shrink > 1 else (st + max_im_shrink) / 2
                flops_det3 = calc_flops(net, img, bt, flip=False)
                if max_im_shrink > 1.5:
                    flops_det3 += calc_flops(net, img, 1.5, flip=False)
                if max_im_shrink > 2:
                    bt *= 2
                    while bt < max_im_shrink:
                        flops_det3 += calc_flops(net, img, bt, flip=False)
                        bt *= 2
                    flops_det3 += calc_flops(net, img, max_im_shrink, flip=False)
                # det4
                flops_det4 = calc_flops(net, img, 0.25, flip=False)
                st = [1.25, 1.75, 2.25]
                for i in range(len(st)):
                    if (st[i] <= max_im_shrink):
                        flops_det4 += calc_flops(net, img, st[i], flip=False)
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