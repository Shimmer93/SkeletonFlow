import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io.video import read_video
from glob import glob
from random import randint

# Not finished

def read_skeletons(skl_fn):
    skls = np.load(skl_fn, allow_pickle=True).item()['rgb_body0']
    return skls

def skeletons_to_flow(skl1, skl2, height, width):
    movement = skl2 - skl1
    flow = torch.zeros((height, width, 2))
    for j, m in zip(skl1, movement):
        flow[int(j[1]), int(j[0])] = m
    flow = flow.permute(2, 0, 1)
    return flow

class NTU60Dataset(Dataset):
    def __init__(self, video_path, skl_path, frame_tsfm, all_tsfm, mode='train', split='xsub'):
        super().__init__()

        self.video_path = video_path
        self.skl_path = skl_path
        self.frame_tsfm = frame_tsfm
        self.all_tsfm = all_tsfm

        self.video_fns = glob(video_path + '/*.avi')
        sample_frames, _, sample_metadata = read_video(self.video_fns[0])
        self.height, self.width = sample_frames.shape[1:3]
        self.fps = sample_metadata['video_fps']

    def __getitem__(self, idx):
        video_fn = self.video_fns[idx]
        frames, _, _ = read_video(video_fn, pts_unit='sec')
        total_frames = frames.shape[0]
        start_frame = randint(0, total_frames - 2)
        frames = frames[start_frame:start_frame + 1].numpy()

        skl_fn = video_fn.replace(self.video_path, self.skl_path).replace('.avi', '.skeleton.npy')
        skls = read_skeletons(skl_fn)
        skls = skls[start_frame:start_frame + 1]

        frames, skls, frame_size = self.all_tsfm(frames, skls)
        frames = self.frame_tsfm(frames)
        flow = skeletons_to_flow(skls[0], skls[1], frame_size[0], frame_size[1])

        return frames, flow

    def __len__(self):
        return len(self.video_fns)