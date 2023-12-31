# Flow visualization code used from https://github.com/tomrunia/OpticalFlow_Visualization


# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-08-03

import numpy as np

def denormalize(img):
    # denormalize an image based on ImageNet stats
    # img = img * np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    # img = img + np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    img = img * 255
    return img.astype(np.uint8)

def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)

def generate_random_colors(num_colors):
    """
    Generate random colors for drawing masks.

    Args:
        num_colors (int): Number of random colors to generate.

    Returns:
        np.ndarray: Random colors of shape [num_colors, 3]
    """
    colors = np.random.randint(0, 256, (num_colors, 3), dtype=np.uint8)
    return colors

def get_predefined_colors(num_kps):
    color_list = \
        [[ 34, 74, 243], [197, 105, 1], [23, 129, 240], [188, 126, 228], [115, 121, 232], [142, 144, 20], 
         [126, 250, 110], [217, 132, 212], [81, 191, 65], [103, 227, 95], [163, 179, 130], [120, 102, 117],
         [199, 85, 111], [98, 251, 87], [59, 24, 47], [55, 244, 124], [251, 221, 136], [186, 25, 19], 
         [172, 81, 95], [96, 76, 118], [11, 43, 76], [181, 55, 80], [157, 186, 192], [80, 185, 205],
         [12, 94, 115], [30, 220, 233], [144, 67, 163], [125, 159, 138], [136, 210, 185], [235, 25, 213]]
    
    colors = np.array(color_list[:num_kps])
    return colors

def mask_to_image(mask, num_kps, convert_to_bgr=False):
    """
    Expects a two dimensional mask image of shape.

    Args:
        mask (np.ndarray): Mask image of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Mask visualization image of shape [H,W,3]
    """
    assert mask.ndim == 2, 'input mask must have two dimensions'
    canvas = np.ones((mask.shape[0], mask.shape[1], 3), np.uint8) * 255
    colors = get_predefined_colors(num_kps)
    for i in range(num_kps):
        canvas[mask == i+1] = colors[i]
    if convert_to_bgr:
        canvas = canvas[...,::-1]
    return canvas

def mask_to_joint_images(masks, num_kps, convert_to_bgr=False):
    assert masks.ndim == 3, 'input mask must have three dimensions'
    assert masks.shape[0] == num_kps, 'input mask must have shape [num_kps, H, W]'
    canvas = np.ones((num_kps, masks.shape[1], masks.shape[2], 3), np.uint8) * 255
    colors = get_predefined_colors(num_kps)
    for i in range(num_kps):
        canvas[i, masks[i] == 1] = colors[i]
    if convert_to_bgr:
        canvas = canvas[...,::-1]
    return canvas
