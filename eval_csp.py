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
    from models import CSP
    net = CSP()
    return net

def calc_shrink(h, w):
    max_im_shrink = (0x7fffffff / 577.0 / (img.shape[0] * img.shape[1])) ** 0.5  # the max size of input image
    shrink = max_im_shrink if max_im_shrink < 1 else 1
    return shrink, max_im_shrink


def calc_flops(net, img, scale=1, flip=False, max_downsample=16):
    img_h, img_w = img.shape[:2]
    img_h_new, img_w_new = int(np.ceil(scale * img_h / 16) * 16), int(np.ceil(scale * img_w / 16) * 16)
    scale_h, scale_w = img_h_new / img_h, img_w_new / img_w

    img_s = cv2.resize(img, None, None, fx=scale_w, fy=scale_h, interpolation=cv2.INTER_LINEAR)

    img_s_h, img_s_w, img_s_c = img_s.shape
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

    st = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0, 2.25]

    img_cnt = 0
    img_total = get_set_size(ANNOTATION)
    flops_avg = 0
    flops_total = int(0)
    # cnt_s = [0] * 5
    with open(ANNOTATION, 'r') as annofile:
        for line in annofile:
            if '.jpg' in line:
                # read the image
                imgpath = osp.join(IMGROOT, line.strip()) # the name of the first image: /0--Parade/0_Parade_marchingband_1_737.jpg
                img = cv2.imread(imgpath) # img.shape <- [height, width, channel]

                # calculate shrink
                shrink, max_im_shrink = calc_shrink(img.shape[0], img.shape[1])
                flops = int(0)

                flops = calc_flops(net, img, 1, flip=False)
                flops = int(0)
                # run profile at set ratio
                for scale in st:
                    if scale in [0.25, 0.5, 0.75, 1]:
                        flops += calc_flops(net, img, scale, flip=True)
                        # pass
                    elif scale <= max_im_shrink:
                        flops += calc_flops(net, img, scale, flip=True)
                        # if scale == 1.25: i = 0
                        # elif scale == 1.50: i = 1
                        # elif scale == 1.75: i = 2
                        # elif scale == 2.00: i = 3
                        # else: i = 4
                        # cnt_s[i] += 1

                # statistics
                img_cnt += 1
                flops_total += flops
                flops_avg = int(flops_total / img_cnt)
                # simple progress bar
                sys.stdout.write('\r')
                sys.stdout.write('-> Profiling model {:s}:: {:d}/{:d}, avg FLOPs: {:,d}, total FLOPs: {:,d}'.format(net.name, img_cnt, img_total, flops_avg, flops_total))
                sys.stdout.flush()
    print('\n')
    # print(cnt_s)