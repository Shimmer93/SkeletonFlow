import torch

def get_min_max_from_skeletons(skls, H, W, padding=10):
    y_mins = torch.min(skls[..., 1], dim=1)[0].type(torch.int)
    y_maxs = torch.max(skls[..., 1], dim=1)[0].type(torch.int)
    x_mins = torch.min(skls[..., 0], dim=1)[0].type(torch.int)
    x_maxs = torch.max(skls[..., 0], dim=1)[0].type(torch.int)

    y_mins = torch.clamp(y_mins - padding, min=0)
    y_maxs = torch.clamp(y_maxs + padding, max=H)
    x_mins = torch.clamp(x_mins - padding, min=0)
    x_maxs = torch.clamp(x_maxs + padding, max=W)
    
    return y_mins, y_maxs, x_mins, x_maxs