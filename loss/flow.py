import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

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
    def __init__(self, flow_temp=0.1, radius=0.7, kernel_size=3, stride=2, loss_weight_mode='norm',
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

def EPE(flow_pred, flow_true, mask=None, real=False):

    if real:
        batch_size, _, h, w = flow_true.shape
        flow_pred = F.interpolate(flow_pred, (h, w), mode='bilinear', align_corners=False)
    else:
        batch_size, _, h, w = flow_pred.shape
        flow_true = F.interpolate(flow_true, (h, w), mode='area')
    if mask != None:
        return torch.norm(flow_pred - flow_true, 2, 1, keepdim=True)[mask].mean()
    else:
        return torch.norm(flow_pred - flow_true, 2, 1).mean()


def EPE_all(flows_pred, flow_true, mask=None, weights=(0.005, 0.01, 0.02, 0.08, 0.32)):

    loss = 0

    for i in range(len(weights)):
        loss += weights[i] * EPE(flows_pred[i], flow_true, mask, real=False)

    return loss

def AAE(flow_pred, flow_true):
    batch_size, _, h, w = flow_true.shape
    flow_pred = F.interpolate(flow_pred, (h, w), mode='bilinear', align_corners=False)
    numerator = torch.sum(torch.mul(flow_pred, flow_pred), dim=1) + 1
    denominator = torch.sqrt(torch.sum(flow_pred ** 2, dim=1) + 1) * torch.sqrt(torch.sum(flow_true ** 2, dim=1) + 1)
    result = torch.clamp(torch.div(numerator, denominator), min=-1.0, max=1.0)

    return torch.acos(result).mean()

def charbonnier(x, alpha=0.25, epsilon=1.e-9):
    return torch.pow(torch.pow(x, 2) + epsilon**2, alpha)


def smoothness_loss(flow):
    b, c, h, w = flow.size()
    v_translated = torch.cat((flow[:, :, 1:, :], torch.zeros(b, c, 1, w, device=flow.device)), dim=-2)
    h_translated = torch.cat((flow[:, :, :, 1:], torch.zeros(b, c, h, 1, device=flow.device)), dim=-1)
    s_loss = charbonnier(flow - v_translated) + charbonnier(flow - h_translated)
    s_loss = torch.sum(s_loss, dim=1) / 2

    return torch.sum(s_loss)/b


def photometric_loss(warped, frm0):
    h, w = warped.shape[2:]
    frm0 = F.interpolate(frm0, (h, w), mode='bilinear', align_corners=False)
    p_loss = charbonnier(warped - frm0)
    p_loss = torch.sum(p_loss, dim=1)/3
    return torch.sum(p_loss)/frm0.size(0)


def unsup_loss(pred_flows, warped_imgs, frm0, weights=(0.005, 0.01, 0.02, 0.08, 0.32)):
    bce = 0
    smooth = 0
    for i in range(len(weights)):
        bce += weights[i] * photometric_loss(warped_imgs[i], frm0)
        smooth += weights[i] * smoothness_loss(pred_flows[i])

    loss = bce + smooth
    return loss, bce, smooth

class EPELoss(nn.Module):
    def __init__(self, gamma):
        super(EPELoss, self).__init__()
        self.gamma = gamma

    def forward(self, pred_flows, flow, mask):
        n_iters = len(pred_flows)
        weights = [self.gamma ** (n_iters - i - 1) for i in range(n_iters)]
        loss = EPE_all(pred_flows, flow, mask, weights)
        return loss

class UnsupLoss(nn.Module):
    def __init__(self, gamma):
        super(UnsupLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pred_flows, frms):
        frm0, frm1 = torch.chunk(frms, 2, dim=1)
        frm0.squeeze_(dim=1)
        frm1.squeeze_(dim=1)
        warped_frms = [self._stn(flow, frm1) for flow in pred_flows]

        n_iters = len(pred_flows)
        weights = [self.gamma ** (n_iters - i - 1) for i in range(n_iters)]
        loss = unsup_loss(pred_flows, warped_frms, frm0, weights)[0]

        return loss
    
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
    
class DisLoss(nn.Module):
    def __init__(self, gamma):
        super(DisLoss, self).__init__()
        self.gamma = gamma
    
    def forward(self, pred_flows, skls, flow):
        n_iters = len(pred_flows)
        weights = [self.gamma ** (n_iters - i - 1) for i in range(n_iters)]
        loss = 0
        for i in range(len(pred_flows)):
            loss += weights[i] * self._dis_loss(pred_flows[i], skls, flow)
        return loss

    def _dis2kp(self, flow, kp):
        # flow: B, 2, H, W
        # kp: B, D

        # print('kp', kp.shape)

        B, _, H, W = flow.shape
        iy, ix = torch.meshgrid(torch.arange(H), torch.arange(W))
        iy = iy.repeat(B, 1, 1).to(flow.device)
        ix = ix.repeat(B, 1, 1).to(flow.device)

        kp_x = kp[..., 0].unsqueeze(-1).unsqueeze(-1)
        kp_y = kp[..., 1].unsqueeze(-1).unsqueeze(-1)

        dis_sq = (ix - kp_x)**2 + (iy - kp_y)**2

        return dis_sq
    
    def _dis2kps(self, flow, kps):
        # flow: B, 2, H, W
        # kps: B, J, D

        # print('kps', kps.shape)

        dis_sqs = []
        for i in range(kps.size(1)):
            dis_sq = self._dis2kp(flow, kps[:, i, :])
            dis_sqs.append(dis_sq)
        dis_sqs = torch.stack(dis_sqs, dim=0)
        dis_sq = torch.min(dis_sqs, dim=0)[0]

        return dis_sq
    
    def _dis_loss(self, pred_flow, kps, flow):
        # flow: B, 2, H, W
        # kps: B, J, D

        dis_sq = self._dis2kps(flow, kps)
        dis = torch.sqrt(dis_sq)
        dis_normed = dis / torch.max(dis)

        # print('dis_normed', dis_normed.unsqueeze(1).shape)
        # print('pred_flow', pred_flow.shape)
        mag = torch.norm(pred_flow, dim=1) ** 2
        dis_loss = torch.mean(dis_normed.unsqueeze(1) * mag)
        return dis_loss
    
class ConLoss(nn.Module):
    def __init__(self, gamma):
        super(ConLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pred_flows, skls, flow):
        n_iters = len(pred_flows)
        weights = [self.gamma ** (n_iters - i - 1) for i in range(n_iters)]
        loss = 0
        for i in range(len(pred_flows)):
            loss += weights[i] * self._con_loss(pred_flows[i], skls, flow)
        return loss
    
    def _sim2kp_flow(self, pred_flow, kp, flow):
        # pred_flow: B, 2, H, W
        # kp: B, D
        # flow: B, 2, H, W
        B = pred_flow.size(0)
        kp_flow = flow[torch.arange(B), :, kp[:, 1].type(torch.int), kp[:, 0].type(torch.int)].unsqueeze(-1).unsqueeze(-1)

        sim = torch.sum(pred_flow * kp_flow, dim=1) / ((torch.norm(pred_flow, dim=1) * torch.norm(kp_flow, dim=1)) + 1e-6)
        return sim
    
    def _sim2flow(self, pred_flow, kps, flow):
        # pred_flow: B, 2, H, W
        # kps: B, J, D
        # flow: B, 2, H, W

        sims = []
        for i in range(kps.size(1)):
            sim = self._sim2kp_flow(pred_flow, kps[:, i, :], flow)
            sims.append(sim)
        sims = torch.stack(sims, dim=0)
        sim = torch.max(sims, dim=0)[0]
        return sim
    
    def _con_loss(self, pred_flow, kps, flow):
        # pred_flow: B, 2, H, W
        # kps: B, J, D
        # flow: B, 2, H, W

        sim = self._sim2flow(pred_flow, kps, flow)
        mag = torch.norm(pred_flow, dim=1) ** 2
        con_loss = -mag * sim
        return con_loss.mean()
    
class ConSegLoss(nn.Module):
    def __init__(self, gamma, thres_con=0.95):
        super(ConSegLoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.thres_con = thres_con

    def forward(self, masks, pred_flows, skls, flows):
        n_iters = len(pred_flows)
        weights = [self.gamma ** (n_iters - i - 1) for i in range(n_iters)]
        loss = 0
        mcons = []
        for i in range(len(pred_flows)):
            l, m = self._con_loss(masks, pred_flows[i], skls, flows)
            loss += weights[i] * l
            mcons.append(m)
        return loss, mcons[-1]

    def _dis2kp(self, flow, kp):
        # flow: B, 2, H, W
        # kp: B, D

        # print('kp', kp.shape)

        B, _, H, W = flow.shape
        iy, ix = torch.meshgrid(torch.arange(H), torch.arange(W))
        iy = iy.repeat(B, 1, 1).to(flow.device)
        ix = ix.repeat(B, 1, 1).to(flow.device)

        kp_x = kp[..., 0].unsqueeze(-1).unsqueeze(-1)
        kp_y = kp[..., 1].unsqueeze(-1).unsqueeze(-1)

        dis_sq = (ix - kp_x)**2 + (iy - kp_y)**2

        return dis_sq
    
    def _dis2whichkp(self, flow, kps):
        # flow: B, 2, H, W
        # kps: B, J, D

        # print('kps', kps.shape)

        dis_sqs = []
        for i in range(kps.size(1)):
            dis_sq = self._dis2kp(flow, kps[:, i, :])
            dis_sqs.append(dis_sq)
        dis_sqs = torch.stack(dis_sqs, dim=1)
        which_kp = torch.argmin(dis_sqs, dim=1, keepdim=True)

        return which_kp

    def _sim2kp_flow(self, pred_flow, kp, flow):
        # pred_flow: B, 2, H, W
        # kp: B, D
        # flow: B, 2, H, W
        B = pred_flow.size(0)
        kp_flow = flow[torch.arange(B), :, kp[:, 1].type(torch.int), kp[:, 0].type(torch.int)].unsqueeze(-1).unsqueeze(-1)

        sim = torch.sum(pred_flow * kp_flow, dim=1) / ((torch.norm(pred_flow, dim=1) * torch.norm(kp_flow, dim=1)) + 1e-6)
        return sim
    
    def _sim2flow(self, pred_flow, kps, flow):
        # pred_flow: B, 2, H, W
        # kps: B, J, D
        # flow: B, 2, H, W

        which_kp = self._dis2whichkp(flow, kps) # B, 1, H, W
        sims = []
        for i in range(kps.size(1)):
            sim = self._sim2kp_flow(pred_flow, kps[:, i, :], flow)
            sims.append(sim)
        sims = torch.stack(sims, dim=1)
        sim_selected = torch.gather(sims, dim=1, index=which_kp)
        return sim_selected

    def _con_loss(self, mask, pred_flow, kps, flow):
        # pred_flow: B, 2, H, W
        # kps: B, J, D
        # flow: B, 2, H, W

        sim = self._sim2flow(pred_flow, kps, flow)
        mask_gt = (sim > self.thres_con).float()
        con_loss = self.criterion(mask, mask_gt)
        return con_loss, mask_gt

class PointSegLoss(nn.Module):
    def __init__(self):
        super(PointSegLoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, masks, skls, gt_masks):
        # masks: B, H, W
        # skls: B, J, D
        # gt_masks: B, H, W

        # print(masks.shape, skls.shape, flows.shape)

        h, w = masks.size(1), masks.size(2)

        flow_masks = (gt_masks > 0)

        y_mins = torch.min(skls[:, :, 1], dim=1)[0].type(torch.int)
        y_maxs = torch.max(skls[:, :, 1], dim=1)[0].type(torch.int)
        x_mins = torch.min(skls[:, :, 0], dim=1)[0].type(torch.int)
        x_maxs = torch.max(skls[:, :, 0], dim=1)[0].type(torch.int)

        # print(y_mins.shape)

        neg_masks = []
        for(i, (y_min, y_max, x_min, x_max)) in enumerate(zip(y_mins, y_maxs, x_mins, x_maxs)):
            num_samples = flow_masks[i].sum()
            choices = torch.arange(0, h*w).view(h, w)
            choices[y_min:y_max, x_min:x_max] = -1
            choices = choices[choices != -1].flatten()
            idxs = torch.randperm(len(choices))[:num_samples]
            samples = choices[idxs]
            canvas = torch.zeros(h*w).to(skls.device)
            canvas[samples] = 1
            canvas = canvas.view(1, h, w)
            neg_masks.append(canvas)
        neg_masks = torch.cat(neg_masks, dim=0).to(dtype=torch.bool)

        # print(f'masks: {masks.shape}, flow_masks: {flow_masks.shape}, neg_masks: {neg_masks.shape}')

        pos_preds = masks[flow_masks]
        pos_gts = torch.ones_like(pos_preds)
        neg_preds = masks[neg_masks]
        neg_gts = torch.zeros_like(neg_preds)

        pos_loss = self.criterion(pos_preds, pos_gts)
        neg_loss = self.criterion(neg_preds, neg_gts)

        loss = pos_loss + neg_loss

        return loss
    
class PointSegLoss2(nn.Module):
    def __init__(self):
        super(PointSegLoss2, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, masks, skls, gt_masks):
        # masks: B, J, H, W
        # skls: B, J, D
        # gt_masks: B, H, W

        # print(masks.shape, skls.shape, flows.shape)
        h, w = masks.size(2), masks.size(3)

        flow_masks = (gt_masks > 0)

        # y_mins = torch.min(skls[:, :, 1], dim=1)[0].type(torch.int)
        # y_maxs = torch.max(skls[:, :, 1], dim=1)[0].type(torch.int)
        # x_mins = torch.min(skls[:, :, 0], dim=1)[0].type(torch.int)
        # x_maxs = torch.max(skls[:, :, 0], dim=1)[0].type(torch.int)

        # # print(y_mins.shape)

        # neg_masks = []
        # for(i, (y_min, y_max, x_min, x_max)) in enumerate(zip(y_mins, y_maxs, x_mins, x_maxs)):
        #     choices = torch.arange(0, h*w).view(h, w)
        #     choices[y_min:y_max, x_min:x_max] = -1
        #     choices = choices[choices != -1].flatten()
        #     idxs = torch.randperm(len(choices))[:10]
        #     samples = choices[idxs]
        #     canvas = torch.zeros(h*w).to(skls.device)
        #     canvas[samples] = 1
        #     canvas = canvas.view(1, h, w)
        #     neg_masks.append(canvas)
        # neg_masks = torch.cat(neg_masks, dim=0).to(dtype=torch.bool)

        # print(f'gt_masks: {gt_masks.shape}, flow_masks: {flow_masks.shape}, neg_masks: {neg_masks.shape}')

        masks_ = masks.transpose(0, 1)
        pos_preds = masks_[:, flow_masks].transpose(0, 1)
        pos_gts = gt_masks[flow_masks].long() - 1
        # neg_preds = masks_[:, neg_masks].transpose(0, 1)
        # neg_gts = torch.zeros_like(neg_preds[:, 0])
        # preds = torch.cat([pos_preds, neg_preds], dim=0)
        # gts = torch.cat([pos_gts, neg_gts], dim=0).long()

        # for i in range(masks.size(1)):
        #     print(f'{len(gts[gts==i])}_{len(preds[preds==i])}', end=' ')
        # print()

        # print(f'pos_preds: {pos_preds.shape}, pos_gts: {pos_gts.shape}, neg_preds: {neg_preds.shape}, neg_gts: {neg_gts.shape}')

        loss = self.criterion(pos_preds, pos_gts)

        return loss

if __name__ == '__main__':
    gamma = 0.8
    n_iters = 12
    pred_flows = [torch.randn(4, 2, 256, 256) for i in range(n_iters)]
    flow = torch.randn(4, 2, 256, 256)
    skls = torch.randn(4, 17, 2)

    flow_mag = torch.sum(flow**2, dim=1, keepdim=True)
    mask = (flow_mag != 0)

    l1 = EPELoss(gamma)
    l2 = UnsupLoss(gamma)
    l3 = DisLoss(gamma)
    l4 = ConLoss(gamma)

    loss1 = l1(pred_flows, flow, mask)
    loss2 = l2(pred_flows, skls)
    loss3 = l3(pred_flows, skls, flow)
    loss4 = l4(pred_flows, skls, flow)

    print(loss1)
    print(loss2)
    print(loss3)
    print(loss4)