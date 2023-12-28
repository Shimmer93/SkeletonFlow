import cv2
import torch
import numpy as np
import torchvision.transforms.functional as TF

def read_image(img_fn):
    img = cv2.imread(img_fn)
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    return img

def add_inter_kp_in_skl(skl, adj_pairs, n_inter=1):
    types = torch.arange(1, skl.shape[0]+1).unsqueeze(1).to(skl.device)
    for pair in adj_pairs:
        i, j = pair[0], pair[1]
        for k in range(n_inter):
            new_kp = (skl[i] * (k+1) + skl[j] * (n_inter-k)) / (n_inter + 1)
            type_ = types[j] if k < n_inter // 2 else types[i]
            types = torch.cat([types, torch.tensor([[type_]], device=skl.device)], dim=0)
            skl = torch.cat([skl, new_kp.unsqueeze(0)], dim=0)
    skl = torch.cat([skl, types], dim=1)
    return skl

def skeletons_to_flow(skl1, skl2, height, width):
    movement = skl2 - skl1
    flow = torch.zeros((height, width, 2))
    for j, m in zip(skl1, movement):
        flow[int(j[1]), int(j[0])] = m[:2]
    flow = flow.permute(2, 0, 1)
    return flow

def skeleton_to_mask2(skl, height, width):
    mask = torch.zeros((height, width))
    for p in skl:
        mask[int(p[1]), int(p[0])] = int(p[2])
    return mask

def draw_line(canvas, start, end, value=1, overlength=0):
    # canvas: torch.Tensor of shape (height, width)
    # start: torch.Tensor of shape (2,)
    # end: torch.Tensor of shape (2,)

    # Bresenham's line algorithm
    # https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
    h, w = canvas.size(0), canvas.size(1)
    x0, y0 = int(start[0]), int(start[1])
    x1, y1 = int(end[0]), int(end[1])
    if overlength > 0:
        dx = x1 - x0
        dy = y1 - y0
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            x0 = int(np.clip(x0 - dx * overlength, 0, h-1))
            y0 = int(np.clip(y0 - dy * overlength, 0, w-1))
            x1 = int(np.clip(x1 + dx * overlength, 0, h-1))
            y1 = int(np.clip(y1 + dy * overlength, 0, w-1))

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while x0 >= 0 and x0 < w and y0 >= 0 and y0 < h:
        canvas[y0, x0] = value
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

def skeleton_to_body_mask(skl, body_parts, height, width):
    mask = torch.zeros((height, width))
    for i, (_, pairs) in enumerate(body_parts.items()):
        for pair in pairs:
            start = skl[pair[0]][:2]
            end = skl[pair[1]][:2]
            draw_line(mask, start, end, value=1, overlength=0.2)
    mask = TF.gaussian_blur(mask.unsqueeze(0), kernel_size=3).squeeze(0)
    mask = (mask > 0).float()
    return mask

def skeleton_to_joint_mask(skl, height, width):
    mask = torch.zeros((skl.shape[0], height, width))
    for i, pt in enumerate(skl):
        canvas = torch.zeros((height, width))
        canvas[int(pt[1]), int(pt[0])] = 1
        canvas = TF.gaussian_blur(canvas.unsqueeze(0), kernel_size=5).squeeze(0)
        canvas = (canvas > 0).float()
        mask[i] = canvas
    return mask