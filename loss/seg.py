import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class BodySegLoss(nn.Module):
    def __init__(self):
        super(BodySegLoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, skls, masks, gt_masks):
        # skls: B J 2
        # masks: B, H, W
        # gt_masks: B, H, W

        h, w = masks.size(1), masks.size(2)

        pos_masks = (gt_masks > 0)

        neg_masks = (gt_masks == 0)

        # y_mins = torch.min(skls[:, :, 1], dim=1)[0].type(torch.int)
        # y_maxs = torch.max(skls[:, :, 1], dim=1)[0].type(torch.int)
        # x_mins = torch.min(skls[:, :, 0], dim=1)[0].type(torch.int)
        # x_maxs = torch.max(skls[:, :, 0], dim=1)[0].type(torch.int)

        # y_mins = torch.clamp(y_mins - 10, min=0)
        # y_maxs = torch.clamp(y_maxs + 10, max=h)
        # x_mins = torch.clamp(x_mins - 10, min=0)
        # x_maxs = torch.clamp(x_maxs + 10, max=w)

        # neg_masks = torch.ones_like(gt_masks, dtype=torch.bool)
        # for i in range(masks.size(0)):
        #     neg_masks[i, y_mins[i]:y_maxs[i], x_mins[i]:x_maxs[i]] = 0

        # neg_masks = []
        # for pos_mask in pos_masks:
        #     num_samples = pos_mask.sum()
        #     choices = torch.arange(0, h*w).view(h, w)
        #     choices[pos_mask] = -1
        #     choices = choices[choices != -1].flatten()
        #     idxs = torch.randperm(len(choices))[:num_samples]
        #     samples = choices[idxs]
        #     canvas = torch.zeros(h*w).to(masks.device)
        #     canvas[samples] = 1
        #     canvas = canvas.view(1, h, w)
        #     neg_masks.append(canvas)
        # neg_masks = torch.cat(neg_masks, dim=0).to(dtype=torch.bool)

        pos_preds = masks[pos_masks]
        pos_gts = torch.ones_like(pos_preds)
        neg_preds = masks[neg_masks]
        neg_gts = torch.zeros_like(neg_preds)

        pos_loss = self.criterion(pos_preds, pos_gts)
        neg_loss = self.criterion(neg_preds, neg_gts)

        loss = 0.5 * pos_loss + 0.5 * neg_loss

        return loss
    
class JointRegLoss(nn.Module):
    def __init__(self):
        super(JointRegLoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, dmaps, gt_dmaps):
        # dmaps: B J H W
        # gt_dmaps: B J H W

        loss = self.criterion(dmaps, gt_dmaps)

        return loss

class JointSegLoss(nn.Module):
    def __init__(self):
        super(JointSegLoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, skls, masks, gt_masks):
        # skls: B J 2
        # masks: B J H W
        # gt_masks: B H W

        h, w = masks.size(2), masks.size(3)

        y_mins = torch.min(skls[:, :, 1], dim=1)[0].type(torch.int)
        y_maxs = torch.max(skls[:, :, 1], dim=1)[0].type(torch.int)
        x_mins = torch.min(skls[:, :, 0], dim=1)[0].type(torch.int)
        x_maxs = torch.max(skls[:, :, 0], dim=1)[0].type(torch.int)

        y_mins = torch.clamp(y_mins - 10, min=0)
        y_maxs = torch.clamp(y_maxs + 10, max=h)
        x_mins = torch.clamp(x_mins - 10, min=0)
        x_maxs = torch.clamp(x_maxs + 10, max=w)

        loss = 0
        for i in range(masks.size(1)):
            pos_masks = (gt_masks == i+1)

            neg_masks = torch.zeros_like(gt_masks, dtype=torch.bool)
            for j in range(masks.size(0)):
                neg_masks[j, y_mins[j]:y_maxs[j], x_mins[j]:x_maxs[j]] = 1
            neg_masks = neg_masks ^ pos_masks

            pos_preds = masks[:,i,...][pos_masks]
            pos_gts = torch.ones_like(pos_preds)
            neg_preds = masks[:,i,...][neg_masks]
            neg_gts = torch.zeros_like(neg_preds)
            
            pos_loss = self.criterion(pos_preds, pos_gts)
            neg_loss = self.criterion(neg_preds, neg_gts)

            loss += 0.1 * pos_loss + 0.9 * neg_loss

        return loss
    
def entropy(x, dim=-1):
    return  - torch.sum(F.softmax(x, dim=dim) * F.log_softmax(x, dim=dim), dim=dim)

def distilled_cross_entropy(l, p, tl=1.0, tp=1.0, mask=None):
    # KL divergence
    """
    l: [*, d]
    p: [*, d]
    return: [*]
    """
    if mask is not None:
        l = F.softmax(l / tl * mask, dim=-1) * mask
        p = F.log_softmax(p / tp * mask, dim=-1) * mask
        loss = torch.sum(-l * p, dim=-1)
        return loss
    else:
        l = F.softmax(l / tl, dim=-1)
        p = F.log_softmax(p / tp, dim=-1)
        loss = torch.sum(-l * p, dim=-1)
        return loss


def make_feature_sim_grid(features, kernel_size=9, stride=None, reference_features=None, 
                        saliency_mask=None, feat_center_detach=False, feat_center_smooth=False):
    # features: [N, C, H, W]
    stride = stride or (kernel_size // 3)
    C, H, W = features.shape[1:]

    features = F.normalize(features, dim=1)
    feat_blocks = F.unfold(features, kernel_size=kernel_size, stride=stride)
    
    feat_blocks = rearrange(feat_blocks, 'N (c w) L -> (N L) w c', c=C)
    center_ind = feat_blocks.shape[1] // 2

    if reference_features is None:
        if saliency_mask is None:
            feat_centers = feat_blocks[:, center_ind, :][:, None, :]  # (N L) 1 c
        else:
            # N, H, W -> N, 1, H, W
            if len(saliency_mask.shape) > 1:
                feat_centers = (feat_blocks * saliency_mask).sum(dim=1, keepdim=True)
            else:
                feat_centers = feat_blocks[torch.arange(0, len(feat_blocks), device=feat_blocks.device), saliency_mask, :][:, None, :]
    else:
        assert saliency_mask is None
        reference_features = F.normalize(reference_features, dim=1)
        ref_feat_blocks = F.unfold(reference_features, kernel_size=kernel_size, stride=stride)
        ref_feat_blocks = rearrange(ref_feat_blocks, 'N (c w) L -> (N L) w c', c=C)
        feat_centers = ref_feat_blocks[:, center_ind, :][:, None, :]  # (N L) 1 c
    
    if feat_center_detach:
        feat_centers = feat_centers.detach()
        
    grid = (feat_centers * feat_blocks).sum(dim=-1) # dot product
    if feat_center_smooth:
        idx = torch.arange(0, len(feat_blocks), device=feat_blocks.device)
        grid[idx, saliency_mask] = 0
        grid[idx, saliency_mask] = grid.max(dim=1).values

    grid = rearrange(grid, '(N h w) (k1 k2) -> N h w k1 k2', k1=kernel_size, k2=kernel_size, 
                     h=int((H-1-(kernel_size-1)) / stride + 1), w=int((W-1-(kernel_size-1)) / stride + 1))
    return grid
    
@torch.no_grad()
def make_optical_flow_grid(flow, kernel_size=9, stride=None,
                           static_threshold=0.1, target_size=None, radius=1.0, return_norm_map=False,
                           normalize=True, maximum_norm=None, return_norm_blocks=False, eps=1e-6,
                           saliency_mask=None, device=None):
    # flow: [N, 2, H, W]
    stride = stride or (kernel_size // 3)

    if target_size is not None:
        flow = F.interpolate(flow, target_size, mode='bicubic')
    
    if device is not None:
        flow = flow.to(device)

    H, W = flow.shape[2:]
    if normalize:
        rescale_flow = flow - flow.flatten(2).mean(dim=2)[..., None, None]  # N, 2, 1, 1
    else:
        rescale_flow = flow
    rescale_flow_norm = rescale_flow.norm(dim=1, keepdim=True)
    rescale_flow_norm /= (eps + (rescale_flow_norm.flatten(1).max(dim=1, keepdim=True).values[..., None, None] if maximum_norm is None else maximum_norm.view(-1)[:, None, None, None])).to(rescale_flow_norm.device)
    flow_norm = F.relu(rescale_flow_norm - static_threshold)

    if return_norm_map:
        return flow_norm 

    flow_blocks = F.unfold(flow, kernel_size=kernel_size, stride=stride)
    flow_norm_blocks = F.unfold(flow_norm, kernel_size=kernel_size, stride=stride)
    
    flow_blocks = rearrange(flow_blocks, 'N (c w) L -> (N L) w c', c=2)
    flow_norm_blocks = rearrange(flow_norm_blocks, 'N w L -> (N L) w')
    center_ind = flow_blocks.shape[1] // 2

    if saliency_mask is None:
        flow_centers = flow_blocks[:, center_ind, :][:, None, :]
    else:
        if len(saliency_mask.shape) > 1:
            flow_centers = (flow_blocks * saliency_mask).sum(dim=1, keepdim=True)
        else:
            flow_centers = flow_blocks[torch.arange(0, len(flow_blocks), device=flow_blocks.device), saliency_mask, :][:, None, :]

    weight = torch.exp(- torch.abs(1 - F.cosine_similarity(flow_centers, flow_blocks, dim=2).clamp_(0.0, 1.0)) / radius)
    grid = weight * flow_norm_blocks
    h=int((H-1-(kernel_size-1)) / stride + 1)
    w=int((W-1-(kernel_size-1)) / stride + 1)
    grid = rearrange(grid, '(N h w) (k1 k2) -> N h w k1 k2', k1=kernel_size, k2=kernel_size, h=h, w=w)
                
    if return_norm_blocks:
        return grid, rearrange(flow_norm_blocks, '(N h w) (k1 k2) -> N h w k1 k2', k1=kernel_size, k2=kernel_size, h=h, w=w) 
    else:
        return grid

class FlowLoss(nn.Module):
    
    def __init__(self, flow_temp=0.1, radius=0.7, kernel_size=5, stride=2, loss_weight_mode='norm',
                loss_weight_margin=0.01, static_threshold=0.1):
        super().__init__()
        self.flow_temp = flow_temp
        self.radius = radius
        self.kernel_size = kernel_size
        self.stride = stride
        self.loss_weight_margin = loss_weight_margin
        self.static_threshold = static_threshold
        self.loss_weight_mode = loss_weight_mode

        self.maximum_entropy = entropy(torch.zeros(1, kernel_size ** 2), dim=-1).item()
    
    
    def forward(self, feat, flow, max_norm, ref_feat=None, mask=None):
        H, W = feat.shape[1], feat.shape[2]
        saliency = feat[:, :, :, -1]
        feat = feat[:, :, :, :-1]
        saliency = saliency.detach()
        saliency = saliency[:, None, :, :]
        sal_blocks = F.unfold(saliency, kernel_size=self.kernel_size, stride=self.stride)
        sal_blocks = rearrange(sal_blocks, 'N w L -> (N L) w')
        saliency_mask = sal_blocks.argmax(dim=1)

        flow_grid = make_optical_flow_grid(flow, target_size=(H, W), stride=self.stride, radius=self.radius,
                            normalize=False, kernel_size=self.kernel_size, saliency_mask=saliency_mask,
                            maximum_norm=max_norm, static_threshold=self.static_threshold, device=feat.device)
        h, w = flow_grid.shape[1:3]
        # flow_grid = flow_grid.cuda()
        feat = feat.permute(0, 3, 1, 2) # -> N, C, H, W
        flow_grid = flow_grid.flatten(3) # N, h, w, kh*kw
        feat_grid = make_feature_sim_grid(feat, kernel_size=self.kernel_size, stride=self.stride, saliency_mask=saliency_mask,
                                        reference_features=ref_feat.permute(0, 3, 1, 2) if ref_feat is not None else None)
        # (N, h*w)
        if self.loss_weight_mode == 'norm':
            flow_w = flow_grid.mean(dim=-1).flatten(1)
            flow_w = F.relu(flow_w - self.loss_weight_margin, inplace=True)
            eps = 1e-6
            flow_w.div_(flow_w.sum(dim=1, keepdim=True) + eps)
        elif self.loss_weight_mode == 'entropy':
            assert self.loss_weight_mode == 'entropy'
            flow_w = flow_grid.view(-1, h * w, self.kernel_size ** 2)
            flow_w = F.relu(self.maximum_entropy - entropy(flow_w, dim=-1)) # N, (h, w)
            eps = 1e-6
            flow_w.div_(flow_w.sum(dim=1, keepdim=True) + eps)
        else:
            flow_w = None
        
        if mask is not None:
            mask_blocks = F.unfold(mask, kernel_size=self.kernel_size, stride=self.stride)
            mask_blocks = rearrange(mask_blocks, 'N k (h w) -> N h w k', h=h, w=w)
        else:
            mask_blocks = None

        # (N, h*w)
        flat_corr_loss = distilled_cross_entropy(flow_grid, feat_grid.flatten(3), tl=self.flow_temp, tp=self.flow_temp, mask=mask_blocks).flatten(1) 
        if flow_w is None:
            flat_corr_loss = flat_corr_loss.mean(dim=1)
        else:
            flat_corr_loss = (flat_corr_loss * flow_w).sum(dim=1)

        return torch.nan_to_num(flat_corr_loss).mean()

class FlowGuidedSegLoss(nn.Module):
    def __init__(self):
        super(FlowGuidedSegLoss, self).__init__()
        self.criterion_sup = nn.BCEWithLogitsLoss()
        self.criterion_unsup = FlowLoss()

    def forward(self, masks, feats, flows):

        # masks: B H W
        # feats: B C h w
        # flows: B 2 H W

        flows_norm = flows.norm(dim=1, keepdim=True)
        max_norm = flows_norm.flatten(2).max(dim=2)[0].unsqueeze(1).unsqueeze(1)

        flows_centered = flows - torch.mean(flows, dim=(2, 3), keepdim=True)
        flows_centered_norm = flows_centered.norm(dim=1, keepdim=True)
        masks_flow = flows_centered_norm > 0.1 * max_norm.max().item()

        loss_sup = self.criterion_sup(masks, masks_flow.float().squeeze(1))
        loss_unsup = self.criterion_unsup(feats, flows, max_norm)

        loss = loss_sup + loss_unsup

        return loss
    
class JointConsistLoss(nn.Module):
    def __init__(self):
        super(JointConsistLoss, self).__init__()
        self.criterion = nn.MSELoss()

    def _generate_grid(self, B, H, W, device):
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()
        grid = torch.transpose(grid, 1, 2)
        grid = torch.transpose(grid, 2, 3)
        grid = grid.to(device)
        return grid
    
    def _stn(self, flow, frm):
        b, _, h, w = flow.shape
        frm = F.interpolate(frm, size=(h, w), mode='bilinear', align_corners=True)
        flow = flow.permute(0, 2, 3, 1)

        grid = flow + self._generate_grid(b, h, w, flow.device)

        factor = torch.FloatTensor([[[[2 / w, 2 / h]]]]).to(flow.device)
        grid = grid * factor - 1
        warped_frm = F.grid_sample(frm, grid, align_corners=True)

        return warped_frm

    def forward(self, masks0, masks1, flow):
        # masks0: B J H W
        # masks1: B J H W
        # skls0: B J 2
        # skls1: B J 2
        # flow: B 2 H W

        # skl_disps = (skls1 - skls0).long()

        masks_warped = self._stn(flow, masks1)
        #         masks_warped[j,i,...] = torch.roll(masks_warped[j,i,...], skl_disps[j,i,1].item(), dims=0)
        #         masks_warped[j,i,...] = torch.roll(masks_warped[j,i,...], skl_disps[j,i,0].item(), dims=1)

        loss = self.criterion(masks_warped, masks0)

        return loss
        

if __name__ == '__main__':
    fl = FlowLoss()
    feat = torch.randn(8, 256, 28, 28)
    flow = torch.randn(8, 2, 224, 224)
    max_norm = torch.randn(8, 1, 1, 1)

    loss = fl(feat, flow, max_norm)

    print(loss)