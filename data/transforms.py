from typing import Any
import numpy as np
import numpy.random as npr
import cv2
import torch
import torchvision.transforms as T

def random_hflip(frms, skls, flip_idxs=None, p=0.5):
    # frm: numpy array of shape N H W C
    # skl: numpy array of shape N J D

    if npr.rand() < p:
        frms = np.flip(frms, axis=2)
        skls[..., 0] = frms.shape[2] - skls[..., 0] - 1
        if flip_idxs is not None:
            skls = skls[:, flip_idxs, :]

    return frms, skls

def random_crop(frms, skls, p=0.5):
    # frm: numpy array of shape N H W C
    # skl: numpy array of shape N J D
    
    h, w = frms.shape[1:3]
    x_min = np.maximum(int(np.min(skls[..., 0])), 0)
    x_max = np.minimum(int(np.max(skls[..., 0])) + 1, w)
    y_min = np.maximum(int(np.min(skls[..., 1])), 0)
    y_max = np.minimum(int(np.max(skls[..., 1])) + 1, h)

    c_size_min = np.maximum(x_max - x_min, y_max - y_min)
    c_size_max = np.minimum(h, w)
    if npr.rand() < p:
        c_size = npr.randint(c_size_min, c_size_max + 1)
    else:
        c_size = c_size_max

    x0 = npr.randint(np.maximum(x_max - c_size, 0), np.minimum(x_min, w - c_size) + 1)
    y0 = npr.randint(np.maximum(y_max - c_size, 0), np.minimum(y_min, h - c_size) + 1)
    x1 = x0 + c_size
    y1 = y0 + c_size

    frms = frms[:, y0:y1, x0:x1, :]
    skls[..., 0] -= x0
    skls[..., 1] -= y0

    return frms, skls

def resize(frms, skls, out_size):
    # frm: numpy array of shape N H W C
    # skl: numpy array of shape N J D

    skls[..., 0] *= out_size[0] / frms.shape[2]
    skls[..., 1] *= out_size[1] / frms.shape[1]

    frms = np.stack([cv2.resize(frm, out_size, interpolation=cv2.INTER_CUBIC) for frm in frms], axis=0)

    return frms, skls

class TrainAllTransforms():
    def __init__(self, hparams):
        self.hparams = hparams

    def __call__(self, frms, skls, flip_idxs=None):
        frms, skls = random_hflip(frms, skls, flip_idxs)
        frms, skls = random_crop(frms, skls, p=self.hparams.p_crop)
        frms, skls = resize(frms, skls, self.hparams.input_size)

        frms = torch.stack([T.ToTensor()(frm) for frm in frms], dim=0)
        skls = torch.from_numpy(skls)

        return frms, skls
    
class TrainFrameTransforms():
    def __init__(self, hparams):
        self.hparams = hparams
        self.tsfm = T.Compose([
            # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.RandomApply([T.ColorJitter(brightness=hparams.brightness, contrast=hparams.contrast, saturation=hparams.saturation, hue=hparams.hue)], p=hparams.p_jitter),
            T.RandomApply([T.GaussianBlur(kernel_size=hparams.kernel_size, sigma=hparams.sigma)], p=hparams.p_blur)
        ])

    def __call__(self, frm):
        return self.tsfm(frm)
    
class ValAllTransforms():
    def __init__(self, hparams):
        self.hparams = hparams

    def __call__(self, frms, skls, flip_idxs=None):
        frms = torch.stack([T.ToTensor()(frm) for frm in frms], dim=0)
        skls = torch.from_numpy(skls)

        return frms, skls
    
class ValFrameTransforms():
    def __init__(self, hparams):
        self.hparams = hparams
        # self.tsfm = T.Compose([
        #     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])

    def __call__(self, frm):
        # return self.tsfm(frm)
        return frm