import cv2
import torch

def read_image(img_fn):
    img = cv2.imread(img_fn)
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    return img

def add_inter_kp_in_skl(skl, adj_pairs, n_inter=1):
    for pair in adj_pairs:
        i, j = pair[0], pair[1]
        for k in range(n_inter):
            new_kp = (skl[i] * (k+1) + skl[j] * (n_inter-k)) / (n_inter + 1)  #(skl[i] + skl[j]) / (n_inter + 1) * (k + 1)
            # print((n_inter + 1), (k + 1), skl[i].tolist(), skl[j].tolist(), new_kp.tolist())
            skl = torch.cat([skl, new_kp.unsqueeze(0)], dim=0)
    return skl

def skeletons_to_flow(skl1, skl2, height, width):
    movement = skl2 - skl1
    flow = torch.zeros((height, width, 2))
    for j, m in zip(skl1, movement):
        flow[int(j[1]), int(j[0])] = m
    flow = flow.permute(2, 0, 1)
    return flow