import torch
import torch.nn as nn
import torch.nn.functional as F

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

    if len(flows_pred) < 5:
        weights = [0.005]*len(flows_pred)
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
    if len(pred_flows) < 5:
        weights = [0.005]*len(pred_flows)
    bce = 0
    smooth = 0
    for i in range(len(weights)):
        bce += weights[i] * photometric_loss(warped_imgs[i], frm0)
        smooth += weights[i] * smoothness_loss(pred_flows[i])

    loss = bce + smooth
    return loss, bce, smooth

class EPELoss(nn.Module):
    def __init__(self, weights=(0.005, 0.01, 0.02, 0.08, 0.32)):
        super(EPELoss, self).__init__()
        self.weights = weights

    def forward(self, pred_flows, flow, mask):
        loss = EPE_all(pred_flows, flow, mask, self.weights)
        return loss

class UnsupLoss(nn.Module):
    def __init__(self, weights=(0.005, 0.01, 0.02, 0.08, 0.32)):
        super(UnsupLoss, self).__init__()
        self.weights = weights

    def forward(self, pred_flows, frms):
        frm0, frm1 = torch.split(frms, 2, dim=1)
        warped_frms = [self._stn(flow, frm1) for flow in pred_flows]

        loss = unsup_loss(pred_flows, warped_frms, frm0, self.weights)

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
        warped_frm = F.grid_sample(frm, grid)

        return warped_frm
    
class DisLoss(nn.Module):
    def __init__(self, weights=(0.005, 0.01, 0.02, 0.08, 0.32)):
        super(DisLoss, self).__init__()
        self.weights = weights
    
    def forward(self, pred_flows, skls, flow):
        loss = 0
        for i in range(len(pred_flows)):
            loss += self.weights[i] * self._dis_loss(pred_flows[i], skls, flow)
        return loss

    def _dis2kp(self, flow, kp):
        # flow: B, 2, H, W
        # kp: B, D

        B, _, H, W = flow.shape
        ix, iy = torch.meshgrid(torch.arange(W), torch.arange(H))
        ix = ix.repeat(B, 1, 1).to(flow.device)
        iy = iy.repeat(B, 1, 1).to(flow.device)

        kp_x = kp[..., 0].unsqueeze(-1).unsqueeze(-1)
        kp_y = kp[..., 1].unsqueeze(-1).unsqueeze(-1)

        dis_sq = (ix - kp_x)**2 + (iy - kp_y)**2

        return dis_sq
    
    def _dis2kps(self, flow, kps):
        # flow: B, 2, H, W
        # kps: B, J, D

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

        dis_loss = torch.mean(dis_normed * pred_flow)
        return dis_loss
    
class ConLoss(nn.Module):
    def __init__(self, weights=(0.005, 0.01, 0.02, 0.08, 0.32)):
        super(ConLoss, self).__init__()
        self.weights = weights

    def forward(self, pred_flows, skls, flow):
        loss = 0
        for i in range(len(pred_flows)):
            loss += self.weights[i] * self._con_loss(pred_flows[i], skls, flow)
        return loss
    
    def _sim2kp_flow(self, pred_flow, kp, flow):
        # pred_flow: B, 2, H, W
        # kp: B, D
        # flow: B, 2, H, W

        kp_flow = flow[:, :, kp[1], kp[0]]
        sim = torch.sum(pred_flow * kp_flow, dim=1) / (torch.norm(pred_flow, dim=1) * torch.norm(kp_flow, dim=1))
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
        con_loss = EPE(pred_flow, flow) * sim
        return con_loss