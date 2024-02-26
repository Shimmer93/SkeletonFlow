import torch
def build_flowformer(cfg):
    name = cfg.transformer 
    if name == 'latentcostformer':
        from .LatentCostFormer.transformer import FlowFormer
    else:
        raise ValueError(f"FlowFormer = {name} is not a valid architecture!")
    print(cfg)
    return FlowFormer(cfg)