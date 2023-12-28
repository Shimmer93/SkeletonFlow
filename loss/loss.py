import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from loss.flow import FlowLoss

def get_min_max_from_skeletons(skls, H, W):
    y_mins = torch.min(skls[:, :, 1], dim=1)[0].type(torch.int)
    y_maxs = torch.max(skls[:, :, 1], dim=1)[0].type(torch.int)
    x_mins = torch.min(skls[:, :, 0], dim=1)[0].type(torch.int)
    x_maxs = torch.max(skls[:, :, 0], dim=1)[0].type(torch.int)

    y_mins = torch.clamp(y_mins - 10, min=0)
    y_maxs = torch.clamp(y_maxs + 10, max=H)
    x_mins = torch.clamp(x_mins - 10, min=0)
    x_maxs = torch.clamp(x_maxs + 10, max=W)
    
    return y_mins, y_maxs, x_mins, x_maxs

class BodySegLoss(nn.Module):
    def __init__(self, w_neg=1):
        super(BodySegLoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()
        self.w_neg = w_neg

    def forward(self, skls, masks, gt_masks):
        B, H, W = masks.size()

        y_mins, y_maxs, x_mins, x_maxs = get_min_max_from_skeletons(skls, H, W)

        pos_masks = (gt_masks > 0)
        neg_masks = torch.ones_like(gt_masks, dtype=torch.bool)
        for i in range(B):
            neg_masks[i, y_mins[i]:y_maxs[i], x_mins[i]:x_maxs[i]] = 0

        pos_preds = masks[pos_masks]
        pos_gt = torch.ones_like(pos_preds)
        neg_preds = masks[neg_masks]
        neg_gt = torch.zeros_like(neg_preds)

        pos_loss = self.criterion(pos_preds, pos_gt)
        neg_loss = self.criterion(neg_preds, neg_gt)

        loss = pos_loss + self.w_neg * neg_loss

        return loss
    
class BodySegLoss2(nn.Module):
    def __init__(self):
        super(BodySegLoss2, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, point_logits, point_labels):
        # point_logits: B P
        # point_labels: B P

        loss = self.criterion(point_logits, point_labels)

        return loss 
        
class JointSegLoss2(nn.Module):
    def __init__(self, w_neg=1):
        super(JointSegLoss2, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()
        self.w_neg = w_neg

    def forward(self, point_logits, point_labels):
        # point_logits: B C P
        # point_labels: B P

        loss = 0
        for i in range(point_logits.size(1)):
            preds = point_logits[:,i,...]
            pos_mask = (point_labels == i+1)
            neg_mask = (point_labels != i+1)
            pos_preds = preds[pos_mask]
            pos_gt = torch.ones_like(pos_preds)
            neg_preds = preds[neg_mask]
            neg_gt = torch.zeros_like(neg_preds)
            loss += 0.2 * self.criterion(pos_preds, pos_gt) + 0.8 * self.criterion(neg_preds, neg_gt)
            # preds_ = torch.cat((pos_preds, neg_preds), dim=0)
            # gts_ = torch.cat((pos_gt, neg_gt), dim=0)
            # loss += self.criterion(preds_, gts_)

        return loss
    
class FlowGuidedSegLoss(nn.Module):
    def __init__(self, w_neg=1):
        super(FlowGuidedSegLoss, self).__init__()
        self.criterion_sup = nn.BCEWithLogitsLoss()
        self.criterion_unsup = FlowLoss()
        self.w_neg = w_neg

    def forward(self, feats, skls, flows, masks, unsup=True):
        # feats: B C h w
        # skls: B J 2
        # flows: B 2 H W

        B, _, H, W = flows.size()
        y_mins, y_maxs, x_mins, x_maxs = get_min_max_from_skeletons(skls, H, W)
        neg_masks = torch.ones((B, H, W), dtype=torch.bool)
        for i in range(B):
            neg_masks[i, y_mins[i]:y_maxs[i], x_mins[i]:x_maxs[i]] = 0

        flows_mean = torch.mean(flows, dim=(2, 3), keepdim=True)
        flows_centered = flows - flows_mean
        flows_centered_norm = flows_centered.norm(dim=1, keepdim=True)
        flows_centered_norm_max = flows_centered_norm.flatten(2).max(dim=2)[0].unsqueeze(1).unsqueeze(1)
        masks_flow = flows_centered_norm > 0.3 * flows_centered_norm_max
        masks_flow = masks_flow.squeeze(1)

        pos_preds = masks[masks_flow]
        pos_gt = torch.ones_like(pos_preds)
        neg_preds = masks[neg_masks]
        neg_gt = torch.zeros_like(neg_preds)
        loss = self.criterion_sup(pos_preds, pos_gt) + self.w_neg * self.criterion_sup(neg_preds, neg_gt)

        if unsup:
            flows_norm = flows.norm(dim=1, keepdim=True)
            max_norm = flows_norm.flatten(2).max(dim=2)[0].unsqueeze(1).unsqueeze(1)
            loss += self.criterion_unsup(feats, flows, max_norm)

        return loss
    
class JointSegLoss(nn.Module):
    def __init__(self, w_neg1=2, w_neg2=5):
        super(JointSegLoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()
        self.w_neg1 = w_neg1
        self.w_neg2 = w_neg2
        self.flag = True

    def forward(self, skls, masks, gt_masks):
        # skls: B J 2
        # masks: B J H W
        # gt_masks: B J H W

        # print(skls.shape, masks.shape, gt_masks.shape)

        B, J, H, W = masks.size()

        # if self.flag:
        #     gt_sum = torch.sum(gt_masks, dim=1)
        #     gt_sum = (gt_sum > 0).float()
        #     import matplotlib.pyplot as plt
        #     plt.imsave('gt_sum.png', gt_sum[0].cpu().numpy())
        #     self.flag = False

        y_mins, y_maxs, x_mins, x_maxs = get_min_max_from_skeletons(skls, H, W)

        box_masks = torch.zeros((B, H, W), dtype=torch.bool)
        for j in range(B):
            box_masks[j, y_mins[j]:y_maxs[j], x_mins[j]:x_maxs[j]] = 1

        loss = 0
        for i in range(J):
            pos_masks = (gt_masks[:,i,...] > 0).to(box_masks.device)
            neg_masks1 = (gt_masks.sum(dim=1) > 0).to(box_masks.device) & ~pos_masks
            # neg_masks1 = box_masks & ~pos_masks
            neg_masks2 = ~box_masks

            pos_preds = masks[:,i,...][pos_masks]
            pos_gts = torch.ones_like(pos_preds)
            neg_preds1 = masks[:,i,...][neg_masks1]
            neg_gts1 = torch.zeros_like(neg_preds1)
            neg_preds2 = masks[:,i,...][neg_masks2]
            neg_gts2 = torch.zeros_like(neg_preds2)
            
            pos_loss = self.criterion(pos_preds, pos_gts)
            neg_loss1 = self.criterion(neg_preds1, neg_gts1)
            neg_loss2 = self.criterion(neg_preds2, neg_gts2)

            loss += 2 * pos_loss + self.w_neg1 * neg_loss1 + self.w_neg2 * neg_loss2
        # loss /= J

        return loss

# class JointSegLoss(nn.Module):
#     def __init__(self, w_neg=1):
#         super(JointSegLoss, self).__init__()
#         self.criterion = nn.CrossEntropyLoss()
#         self.w_neg = w_neg

#     def forward(self, masks, gt_masks):
#         # masks: B J H W
#         # gt_masks: B H W
#         B, J, H, W = masks.size()

#         masks_ = F.softmax(masks, dim=1)

#         loss = 0
#         for i in range(B):
#             pos_mask = (gt_masks[i] > 0)
#             neg_mask = (gt_masks[i] == 0)

#             pos_preds = masks_[i, :, pos_mask].unsqueeze(0)
#             pos_gt = gt_masks[i, pos_mask].unsqueeze(0)
#             neg_preds = masks_[i, :, neg_mask].unsqueeze(0)
#             neg_gt = gt_masks[i, neg_mask].unsqueeze(0)

#             pos_loss = self.criterion(pos_preds, pos_gt.long())
#             neg_loss = self.criterion(neg_preds, neg_gt.long())
#             loss += pos_loss + self.w_neg * neg_loss
#         loss /= B
#         # print(masks.shape, gt_masks.shape)
#         # loss = self.criterion(masks, gt_masks.long())

#         return loss
    
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

        loss = self.criterion(masks_warped, masks0.detach())

        return loss

class WeakSupFlowLoss(nn.Module):
    def __init__(self):
        super(WeakSupFlowLoss, self).__init__()
        self.criterion = nn.CosineSimilarity(dim=0, eps=1e-6)

    def forward(self, flow, gt_flow):
        # flow: B 2 H W
        # gt_flow: B 2 H W

        gt_flow_norm = gt_flow.norm(dim=1)
        # pos_mask = (gt_flow_norm.repeat(1, 2, 1, 1) > 0)
        pos_mask = (gt_flow_norm > 0)
        # pos_pred = flow[pos_mask]
        # pos_gt = gt_flow[pos_mask]
        pos_pred = flow.transpose(0, 1)[:, pos_mask]
        pos_gt = gt_flow.transpose(0, 1)[:, pos_mask]

        loss = -self.criterion(pos_pred, pos_gt).mean()

        return loss