import yaml
from argparse import Namespace
import numpy as np
import torch
import torch.nn.functional as F
import pickle
from PIL import Image

def load_cfg(cfg):
    hyp = None
    if isinstance(cfg, str):
        with open(cfg, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    return Namespace(**hyp)


def merge_args_cfg(args, cfg):
    dict0 = vars(args)
    dict1 = vars(cfg)
    dict = {**dict0, **dict1}

    return Namespace(**dict)

def write_psm(save_path, joint_masks, body_mask=None, obj_mask=None, rescale_ratio=1.0):

    out_masks = joint_masks
    if body_mask is not None:
        out_masks = torch.cat([out_masks, body_mask.unsqueeze(0)], dim=0)
    if obj_mask is not None:
        out_masks = torch.cat([out_masks, obj_mask.unsqueeze(0)], dim=0)
    out_masks = F.interpolate(out_masks.unsqueeze(0), scale_factor=1.0/rescale_ratio, \
                              mode='bilinear', align_corners=False).squeeze(0)
    out_masks = (out_masks > 0.5).cpu().numpy().astype(np.uint8)
    J, H, W = out_masks.shape

    nw = 4
    nh = int(np.ceil(J / nw))
    canvas = np.zeros((H * nh, W * nw), dtype=np.uint8)
    for i in range(J):
        x = (i % nw) * W
        y = (i // nw) * H
        canvas[y:y+H, x:x+W] = out_masks[i]
    canvas[-1, -1] = J
    canvas[-1, -2] = H
    canvas[-1, -3] = W
    
    Image.fromarray(canvas).save(save_path)

def read_psm(psm_path):
    # PSM is stored in a PNG image
    canvas = np.array(Image.open(psm_path)).squeeze()

    # Attributes are stored in the last row
    J = canvas[-1, -1]
    H = canvas[-1, -2]
    W = canvas[-1, -3]
    canvas[-1, -1] = 0
    canvas[-1, -2] = 0
    canvas[-1, -3] = 0
    nw = 4
    nh = int(np.ceil(J / nw))

    # Split the canvas into masks
    out_masks = np.zeros((J, H, W), dtype=np.uint8)
    for i in range(J):
        x = (i % nw) * W
        y = (i // nw) * H
        out_masks[i] = canvas[y:y+H, x:x+W]
    out_masks = torch.from_numpy(out_masks) > 0
    
    return out_masks