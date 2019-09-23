import scipy.io as sio
import os.path as osp

class WIDERFACE(object):
    def __init__(self, widerface_root, eval_tools_root):
        super(WIDERFACE, self).__init__()

        assert osp.isdir(widerface_root)
        assert osp.isdir(eval_tools_root)

        self.widerface_root = widerface_root
        self.eval_tools_root = eval_tools_root

    def _contruct_filepaths_list(self, loc, mode='mat'):
        filepaths = []

        if mode == 'mat':
            mat_loc = osp.join(self.eval_tools_root, loc)
            mat = sio.loadmat(mat_loc)
            assert len(mat['event_list']) == len(mat['file_list'])

            for event_id in range(len(mat['event_list'])):
                for img_id in range(len(mat['file_list'][event_id][0])):
                    filepaths.append(
                        osp.join(
                            # self.widerface_root,
                            # 'WIDER_val',
                            mat['event_list'][event_id][0][0],
                            mat['file_list'][event_id][0][img_id][0][0]
                        )+'.jpg'
                    )
            return filepaths
        else:
            raise NotImplementedError

    @property
    def trainset_filepaths(self):
        raise NotImplementedError

    @property
    def valset_filepaths(self):
        raise NotImplementedError

    @property
    def testset_filepaths(self):
        raise NotImplementedError

    @property
    def valset_easy_filepaths(self):
        return self._contruct_filepaths_list(osp.join('ground_truth', 'wider_easy_val.mat'), mode='mat')

    @property
    def valset_medium_filepaths(self):
        return self._contruct_filepaths_list(osp.join('ground_truth', 'wider_medium_val.mat'), mode='mat')

    @property
    def valset_hard_filepaths(self):
        return self._contruct_filepaths_list(osp.join('ground_truth', 'wider_hard_val.mat'), mode='mat')