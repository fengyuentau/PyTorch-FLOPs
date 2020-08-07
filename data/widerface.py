import os

import cv2
import numpy as np
import scipy.io as io

class WIDERFace(object):
    def __init__(self, widerface_root='./data/widerface', split='val'):
        super(WIDERFace, self).__init__()

        assert os.path.exists(widerface_root), '{} does not exists.'.format(widerface_root)

        split = split.lower()
        splits = ['val-easy', 'val-medium', 'val-hard', 'val', 'test']
        assert any(split==c for c in splits), 'split muse be one of {}, while {} is not one of them.'.format(splits, split)

        self.widerface_root = widerface_root
        self.split = split

        self.eval_tools_root = os.path.join(self.widerface_root, 'ground_truth')
        self.val = {
            'img_root': os.path.join(self.widerface_root, 'WIDER_val', 'images'),
            'anno': {
                'txt': os.path.join(self.widerface_root, 'wider_face_split', 'wider_face_val_bbx_gt.txt'),
                'mat': {
                    'all': os.path.join(self.widerface_root, 'wider_face_split', 'wider_face_val.mat'),
                    'easy': os.path.join(self.eval_tools_root, 'wider_easy_val.mat'),
                    'medium': os.path.join(self.eval_tools_root, 'wider_medium_val.mat'),
                    'hard': os.path.join(self.eval_tools_root, 'wider_hard_val.mat')
                }
            }
            
        }
        self.test = {
            'img_root': os.path.join(self.widerface_root, 'WIDER_test', 'images'),
            'anno': {
                'txt': os.path.join(self.widerface_root, 'wider_face_split', 'wider_face_test_filelist.txt'),
                'mat': {
                    'all': os.path.join(self.widerface_root, 'wider_face_split', 'wider_face_test.mat')
                }
            }
        }

        if self.split in splits[:3]:
            self.filepaths, self.bboxes = self._read_from_mat(self.split)
        elif self.split in splits[3:]:
            self.filepaths, self.bboxes = self._read_from_txt(self.split)

    def _read_from_mat(self, split: str):
        subset = split.split('-')[-1]
        mat_loc = eval('self.'+split)['anno']['mat'][subset]
        assert mat_loc.endswith('.mat'), '{} should be .mat file.'.format(mat_loc)
        assert os.path.exists(mat_loc), '{} does not exists.'.format(mat_loc)

        filepaths = []
        bboxes = []
        parent_path = self.eval('self.'+split)['img_root']
        mat = io.loadmat(mat_loc)
        for event_id in range(len(mat['event_list'])):
            event_name = mat['event_list'][event_id][0][0]

            for img_id in range(len(mat['file_list'][event_id][0])):
                filenames = mat['file_list'][event_id][0][img_id][0][0]
                filepaths.append(
                    os.path.join(parent_path, event_name, filenames+'.jpg')
                )

                # TODO: read bboxes and attributes
                # img_bboxes = []
                # for bbox_id in range(len(mat['face_bbx_list'][event_id][0][img_id][0])):
                #     bbox = mat['face_bbx_list'][event_id][0][img_id][0][bbox_id]
                #     img_bboxes.append(bbox)
                # bboxes.append(img_bboxes)
        return filepaths, bboxes

    def _read_from_txt(self, split: str):
        txt_loc = eval('self.'+split)['anno']['txt']
        assert txt_loc.endswith('.txt'), '{} should be .txt file.'.format(txt_loc)
        assert os.path.exists(txt_loc), '{} does not exists.'.format(txt_loc)

        filepaths = []
        bboxes = []
        parent_path = eval('self.'+split)['img_root']
        with open(txt_loc, 'r') as f:
            for line in f:
                line = line.strip()
                if line.endswith('.jpg'):    # image path
                    filepaths.append(
                        os.path.join(parent_path, line)
                    )

                    # TODO: read bboxes and attributes
                    # img_bboxes = []
                    # nface = int(next(f))
                    # for i in range(nface):
                    #     line = next(f)
                    #     line = line.strip().split()
                    #     img_bboxes.append(
                    #         [int(line[0]), int(line[1]), int(line[2]), int(line[3])]
                    #     )
                    # bboxes.append(img_bboxes)
        return filepaths, bboxes

    def __getitem__(self, index):
        img_loc = self.filepaths[index]
        img = cv2.imread(img_loc)
        return img
        # return self.filepaths[index], self.bboxes[index]

    def __len__(self):
        return len(self.filepaths)


if __name__ == '__main__':
    dataset = WIDERFace()
    print(len(dataset))

    counter = 0
    for img in dataset:
        if counter == 0:
            print(img.shape)
        counter += 1
    print(counter)