import cv2
import torch

def read_image(img_fn):
    img = cv2.imread(img_fn)
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    return img

def skeletons_to_flow(skl1, skl2, height, width):
    movement = skl2 - skl1
    flow = torch.zeros((height, width, 2))
    for j, m in zip(skl1, movement):
        flow[int(j[1]), int(j[0])] = m
    flow = flow.permute(2, 0, 1)
    return flow