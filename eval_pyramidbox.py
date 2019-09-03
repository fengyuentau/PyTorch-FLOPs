import os
import os.path as osp
import argparse
import sys
import time

import numpy as np
from PIL import Image

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
    from models import PyramidBox
    net = PyramidBox([3, 1024, 1024])
    return net

def calc_shrink(height, width):
    """
    Args:
        height (int): image height.
        width (int): image width.
    """
    # avoid out of memory
    max_shrink_v1 = (0x7fffffff / 577.0 / (height * width))**0.5
    max_shrink_v2 = ((678 * 1024 * 2.0 * 2.0) / (height * width))**0.5

    def get_round(x, loc):
        str_x = str(x)
        if '.' in str_x:
            str_before, str_after = str_x.split('.')
            len_after = len(str_after)
            if len_after >= 3:
                str_final = str_before + '.' + str_after[0:loc]
                return float(str_final)
            else:
                return x

    max_shrink = get_round(min(max_shrink_v1, max_shrink_v2), 2) - 0.3
    if max_shrink >= 1.5 and max_shrink < 2:
        max_shrink = max_shrink - 0.1
    elif max_shrink >= 2 and max_shrink < 3:
        max_shrink = max_shrink - 0.2
    elif max_shrink >= 3 and max_shrink < 4:
        max_shrink = max_shrink - 0.3
    elif max_shrink >= 4 and max_shrink < 5:
        max_shrink = max_shrink - 0.4
    elif max_shrink >= 5:
        max_shrink = max_shrink - 0.5

    shrink = max_shrink if max_shrink < 1 else 1
    return shrink, max_shrink


def calc_flops(net, image, shrink=1, flip=True, max_downsample=16):
    image_shape = [3, image.size[1], image.size[0]]
    if shrink != 1:
        h, w = int(image_shape[1] * shrink), int(image_shape[2] * shrink)
        # x = cv2.resize(img, None, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_LINEAR)
        image = image.resize((w, h), Image.ANTIALIAS)
        image_shape = [3, h, w]

    c, h, w = image_shape
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
                # img = cv2.imread(imgpath) # img.shape <- [height, width, channel]
                img = Image.open(imgpath)

                # calculate shrink
                shrink, max_im_shrink = calc_shrink(img.size[1], img.size[0])

                # det0 and det1 (det1 is the flipped version of det0)
                flops_det01 = calc_flops(net, img, shrink, flip=True)
                # det2
                st = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink
                flops_det2 = calc_flops(net, img, st, flip=False)
                # det3
                bt = min(2, max_im_shrink) if max_im_shrink > 1 else (st + max_im_shrink) / 2
                flops_det3 = calc_flops(net, img, bt, flip=False)
                if max_im_shrink > 2:
                    bt *= 2
                    while bt < max_im_shrink:
                        flops_det3 += calc_flops(net, img, bt, flip=False)
                        bt *= 2
                    flops_det3 += calc_flops(net, img, max_im_shrink, flip=False)
                # det4
                flops_det4 = calc_flops(net, img, 0.25, flip=False)
                st = [0.75, 1.25, 1.5, 1.75]
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