import numpy as np
from torch.utils.data import Dataset
from glob import glob
from random import randint
import os
import json

from data.utils import read_image, skeletons_to_flow

class SubJHMDBDataset(Dataset):
    def __init__(self, data_dir, mode='train', split=1, all_tsfm=None, frm_tsfm=None):
        super().__init__()

        self.data_dir = data_dir
        self.mode = mode
        self.split = split
        self.all_tsfm = all_tsfm
        self.frm_tsfm = frm_tsfm
        self.flip_idxs = [0,1,2,4,3,6,5,8,7,10,9,12,11,14,13]

        # train_test = 'train' if mode in ['train', 'val'] else 'test'
        train_test = 'train' if mode == 'train' else 'test'
        with open(os.path.join(data_dir, 'annotations', f'Sub{split}_{train_test}.json'), 'r') as f:
            data = json.load(f)
        self.frm_data = data['images']
        self.skl_data = data['annotations']
        self.idxs = np.random.permutation(len(self.frm_data)) if mode == 'train' else np.arange(len(self.frm_data))

    def __getitem__(self, idx):
        i0 = self.idxs[idx]

        frm0_data = self.frm_data[i0]
        frm0_fn = os.path.join(self.data_dir, frm0_data['file_name'])
        i_frm0 = int(os.path.basename(frm0_fn).split('.')[0])
        n_frms = frm0_data['nframes']
        i_frm1 = i_frm0 + 1 if i_frm0 < n_frms - 1 else i_frm0
        i1 = i0 + 1 if i_frm0 < n_frms - 1 else i0

        frm1_fn = frm0_fn.replace(f'{i_frm0:05d}', f'{i_frm1:05d}')
        frm0 = read_image(frm0_fn)
        frm1 = read_image(frm1_fn)
        frms = np.stack([frm0, frm1], axis=0)

        skl0 = self._get_skeleton(i0)
        skl1 = self._get_skeleton(i1)
        skls = np.stack([skl0, skl1], axis=0)

        frms, skls = self.all_tsfm(frms, skls, self.flip_idxs)
        frms = self.frm_tsfm(frms)
        flow = skeletons_to_flow(skls[0], skls[1], frms.shape[2], frms.shape[3])

        return frms, skls, flow
    
    def __len__(self):
        return len(self.frm_data)

    def _get_skeleton(self, i):
        _skl = np.array(self.skl_data[i]['keypoints']).reshape(-1, 3)
        skl = _skl[..., :2] - 1
        #skl_visible = np.minimum(1, _skl[..., 2])
        return skl
