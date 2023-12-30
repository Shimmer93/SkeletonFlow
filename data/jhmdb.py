import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob
from random import randint
import os
import json

import sys
sys.path.append('/mnt/home/zpengac/USERDIR/HAR/SkeletonFlow')
from data.utils import read_image, skeletons_to_flow, add_inter_kp_in_skl, skeleton_to_body_mask, skeleton_to_joint_mask

class SubJHMDBDataset(Dataset):
    def __init__(self, data_dir, mode='train', split=1, n_inter=2, all_tsfm=None, frm_tsfm=None):
        super().__init__()

        self.data_dir = data_dir
        self.mode = mode
        self.split = split
        self.n_inter = n_inter
        self.all_tsfm = all_tsfm
        self.frm_tsfm = frm_tsfm
        self.flip_idxs = [0,1,2,4,3,6,5,8,7,10,9,12,11,14,13]
        self.adj_pairs = [[0,1],[0,2],[0,3],[0,4],[1,5],[1,6],[3,7],[4,8],[5,9],[6,10],[7,11],[8,12],[9,13],[10,14]]
        self.body_parts = {
            'head': [[0, 2]],
            'torso': [[0, 1], [0, 3], [0, 4], [1, 5], [1, 6], [3, 5], [4, 6], [5, 6]],
            'left upper arm': [[3, 7]],
            'left lower arm': [[7, 11]],
            'right upper arm': [[4, 8]],
            'right lower arm': [[8, 12]],
            'left upper leg': [[5, 9]],
            'left lower leg': [[9, 13]],
            'right upper leg': [[6, 10]],
            'right lower leg': [[10, 14]]
        }
        self.joint_groups = [[0], [1], [2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]]

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
        i_frm1 = i_frm0 + 1 if i_frm0 < n_frms else i_frm0
        i1 = i0 + 1 if i_frm0 < n_frms else i0

        frm1_fn = frm0_fn.replace(f'{i_frm0:05d}', f'{i_frm1:05d}')
        frm0 = read_image(frm0_fn)
        frm1 = read_image(frm1_fn)
        if frm0.shape[0] != frm1.shape[0] or frm0.shape[1] != frm1.shape[1]:
            frm1 = frm0
        frms = np.stack([frm0, frm1], axis=0)

        skl0 = self._get_skeleton(i0)
        skl1 = self._get_skeleton(i1)
        skls = np.stack([skl0, skl1], axis=0)

        frms, skls = self.all_tsfm(frms, skls, self.flip_idxs)
        frms = self.frm_tsfm(frms)
        flow = skeletons_to_flow(skls[0], skls[1], frms.shape[2], frms.shape[3])
        
        mask_body0 = skeleton_to_body_mask(skls[0], self.body_parts, frms.shape[2], frms.shape[3])
        mask_body1 = skeleton_to_body_mask(skls[1], self.body_parts, frms.shape[2], frms.shape[3])
        mask_joint0 = skeleton_to_joint_mask(skls[0], frms.shape[2], frms.shape[3])
        mask_joint1 = skeleton_to_joint_mask(skls[1], frms.shape[2], frms.shape[3])
        masks_body = torch.stack([mask_body0, mask_body1], dim=0)
        masks_joint = torch.cat([mask_joint0, mask_joint1], dim=0)
        masks = torch.cat([masks_body, masks_joint], dim=0)

        return frms, skls, flow, masks, frm0_fn
    
    def __len__(self):
        return len(self.frm_data)

    def _get_skeleton(self, i):
        _skl = np.array(self.skl_data[i]['keypoints']).reshape(-1, 3)
        skl = _skl[..., :2] - 1
        #skl_visible = np.minimum(1, _skl[..., 2])
        return skl
    
if __name__ == '__main__':
    from data.transforms import TrainAllTransforms, TrainFrameTransforms, ValAllTransforms, ValFrameTransforms
    class hparams:
        input_size= [224, 224]
        brightness= 0.4
        contrast= 0.4
        saturation= 0.4
        hue= 0.1
        p_jitter= 0.5
        kernel_size= 3
        sigma= 0.1
        p_blur= 0.5
    d = SubJHMDBDataset('/mnt/home/zpengac/USERDIR/HAR/datasets/jhmdb', mode='test', split=1, n_inter=2, all_tsfm=ValAllTransforms(hparams), frm_tsfm=ValFrameTransforms(hparams))
    frms, skls, flow, masks = d[0]
    print(frms.shape, skls.shape, flow.shape, masks.shape)
    import matplotlib.pyplot as plt
    mask_bin = (masks > 0.5).float()
    plt.imsave('mask0.png', mask_bin[0])
    plt.imsave('mask1.png', mask_bin[1])