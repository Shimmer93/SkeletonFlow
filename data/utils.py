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
            new_kp = (skl[i] * (k+1) + skl[j] * (n_inter-k)) / (n_inter + 1)  #(skl[i] + skl[j]) / (n_inter + 1) * (k + 1)
            # print((n_inter + 1), (k + 1), skl[i].tolist(), skl[j].tolist(), new_kp.tolist())
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
            draw_line(mask, start, end, value=1, overlength=0.1)
    return mask

def skeleton_to_joint_mask(skl, height, width):
    mask = torch.zeros((height, width))
    for i, pt in enumerate(skl):
        canvas = torch.zeros((height, width))
        canvas[int(pt[1]), int(pt[0])] = 1
        canvas = TF.gaussian_blur(canvas.unsqueeze(0), kernel_size=3).squeeze(0)
        canvas = (canvas > 0).float()
        mask = torch.maximum(mask, canvas * (i+1))
    return mask

def skeleton_to_joint_dmap(skl, height, width, kernel_size=7, sigma=4):
    dmap = torch.zeros((skl.shape[0], height, width))
    for i, pt in enumerate(skl):
        dmap[i, int(pt[1]), int(pt[0])] = 1
        dmap[i:i+1] = TF.gaussian_blur(dmap[i:i+1], kernel_size=kernel_size, sigma=sigma)
        dmap[i:i+1] /= dmap[i:i+1].max()
    return dmap

# def skeleton_to_joint_dmap(skl, height, width):
#     mask = torch.zeros((height, width))
#     for i, pt in enumerate(skl):
#         canvas = torch.zeros((height, width))
#         canvas[int(pt[1]), int(pt[0])] = 1
#         canvas = TF.gaussian_blur(canvas.unsqueeze(0), kernel_size=3).squeeze(0)
#         # canvas = (canvas > 0).float()
#         mask = torch.maximum(mask, canvas * (i+1))
#     return mask
    
def generate_a_heatmap(arr, centers, sigma):
    """Generate pseudo heatmap for one keypoint in one frame.

    Args:
        arr (np.ndarray): The array to store the generated heatmaps. Shape: img_h * img_w.
        centers (np.ndarray): The coordinates of corresponding keypoints (of multiple persons). Shape: M * 2.
        max_values (np.ndarray): The max values of each keypoint. Shape: M.

    Returns:
        np.ndarray: The generated pseudo heatmap.
    """

    img_h, img_w = arr.shape

    for center in centers:

        mu_x, mu_y = center[0], center[1]
        st_x = max(int(mu_x - 3 * sigma), 0)
        ed_x = min(int(mu_x + 3 * sigma) + 1, img_w)
        st_y = max(int(mu_y - 3 * sigma), 0)
        ed_y = min(int(mu_y + 3 * sigma) + 1, img_h)
        x = torch.arange(st_x, ed_x, 1, torch.float32)
        y = torch.arange(st_y, ed_y, 1, torch.float32)

        # if the keypoint not in the heatmap coordinate system
        if not (len(x) and len(y)):
            continue
        y = y[:, None]

        patch = torch.exp(-((x - mu_x)**2 + (y - mu_y)**2) / 2 / sigma**2)
        arr[st_y:ed_y, st_x:ed_x] = torch.maximum(arr[st_y:ed_y, st_x:ed_x], patch)

def generate_a_limb_heatmap(arr, starts, ends, sigma):
    """Generate pseudo heatmap for one limb in one frame.

    Args:
        arr (np.ndarray): The array to store the generated heatmaps. Shape: img_h * img_w.
        starts (np.ndarray): The coordinates of one keypoint in the corresponding limbs. Shape: M * 2.
        ends (np.ndarray): The coordinates of the other keypoint in the corresponding limbs. Shape: M * 2.
        start_values (np.ndarray): The max values of one keypoint in the corresponding limbs. Shape: M.
        end_values (np.ndarray): The max values of the other keypoint in the corresponding limbs. Shape: M.

    Returns:
        np.ndarray: The generated pseudo heatmap.
    """

    img_h, img_w = arr.shape

    for start, end in zip(starts, ends):

        min_x, max_x = min(start[0], end[0]), max(start[0], end[0])
        min_y, max_y = min(start[1], end[1]), max(start[1], end[1])

        min_x = max(int(min_x - 3 * sigma), 0)
        max_x = min(int(max_x + 3 * sigma) + 1, img_w)
        min_y = max(int(min_y - 3 * sigma), 0)
        max_y = min(int(max_y + 3 * sigma) + 1, img_h)

        x = torch.arange(min_x, max_x, 1, torch.float32)
        y = torch.arange(min_y, max_y, 1, torch.float32)

        if not (len(x) and len(y)):
            continue

        y = y[:, None]
        x_0 = torch.zeros_like(x)
        y_0 = torch.zeros_like(y)

        # distance to start keypoints
        d2_start = ((x - start[0])**2 + (y - start[1])**2)

        # distance to end keypoints
        d2_end = ((x - end[0])**2 + (y - end[1])**2)

        # the distance between start and end keypoints.
        d2_ab = ((start[0] - end[0])**2 + (start[1] - end[1])**2)

        if d2_ab < 1:
            generate_a_heatmap(arr, start[None], sigma)
            continue

        coeff = (d2_start - d2_end + d2_ab) / 2. / d2_ab

        a_dominate = coeff <= 0
        b_dominate = coeff >= 1
        seg_dominate = 1 - a_dominate - b_dominate

        position = torch.stack([x + y_0, y + x_0], axis=-1)
        projection = start + torch.stack([coeff, coeff], axis=-1) * (end - start)
        d2_line = position - projection
        d2_line = d2_line[:, :, 0]**2 + d2_line[:, :, 1]**2
        d2_seg = a_dominate * d2_start + b_dominate * d2_end + seg_dominate * d2_line

        patch = torch.exp(-d2_seg / 2. / sigma**2)

        arr[min_y:max_y, min_x:max_x] = torch.maximum(arr[min_y:max_y, min_x:max_x], patch)

def skeleton_to_density_map(skl, adj_pairs, height, width, sigma=0.6):
    canvas = torch.zeros((height, width))
    starts = []
    ends = []
    for pair in adj_pairs:
        i, j = pair[0], pair[1]
        starts.append(skl[i][:2])
        ends.append(skl[j][:2])
    generate_a_limb_heatmap(canvas, torch.stack(starts), torch.stack(ends), sigma)
    return canvas