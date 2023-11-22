from typing import Any
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import matplotlib.pyplot as plt
import os

from model.flownet import FlowNetS
from model.raft import RAFT
from model.raft_modified import RAFT_Modified
from loss.flow import EPELoss, UnsupLoss, DisLoss, ConLoss, ConSegLoss, PointSegLoss
from misc.vis import denormalize, flow_to_image

def create_model(hparams):
    if hparams.model_name == 'flownet':
        return FlowNetS()
    elif hparams.model_name == 'raft':
        return RAFT(hparams)
    elif hparams.model_name == 'raft_modified':
        return RAFT_Modified(hparams)
    else:
        raise NotImplementedError

def create_optimizer(hparams, mparams):
    if hparams.optim_name == 'adam':
        return optim.Adam(mparams, lr=hparams.lr, weight_decay=hparams.weight_decay)
    elif hparams.optim_name == 'adamw':
        return optim.AdamW(mparams, lr=hparams.lr, weight_decay=hparams.weight_decay)
    elif hparams.optim_name == 'sgd':
        return optim.SGD(mparams, lr=hparams.lr, momentum=hparams.momentum)
    else:
        raise NotImplementedError
    
def create_scheduler(hparams, optimizer):
    if hparams.sched_name == 'cosine':
        return LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=hparams.warmup_epochs, 
                max_epochs=hparams.epochs, warmup_start_lr=hparams.warmup_lr, eta_min=hparams.min_lr)
    elif hparams.sched_name == 'step':
        return sched.MultiStepLR(optimizer, milestones=hparams.milestones, gamma=hparams.gamma)
    elif hparams.sched_name == 'plateau':
        return sched.ReduceLROnPlateau(optimizer, patience=hparams.patience, factor=hparams.factor, 
                min_lr=hparams.min_lr)
    else:
        raise NotImplementedError

class LitModel(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = create_model(hparams)
        self.sup_loss = EPELoss(hparams.gamma)
        self.unsup_loss = UnsupLoss(hparams.gamma)
        # self.dis_loss = DisLoss(hparams.gamma)
        # self.con_loss = ConLoss(hparams.gamma)
        self.con_loss = ConSegLoss(hparams.gamma, hparams.thres_con)
        self.point_loss = PointSegLoss()

        os.makedirs(f'{self.hparams.model_ckpt_dir}/vis', exist_ok=True)
    
    def _calculate_loss(self, frms, skls, flow, pred_flows, masks=None, mode='train'):
        flow_mag = torch.sum(flow**2, dim=1, keepdim=True)
        mask = (flow_mag != 0)
        l_sup = self.sup_loss(pred_flows, flow, mask)
        l_unsup = self.unsup_loss(pred_flows, frms)
        l_con, mcons = self.con_loss(masks, pred_flows, skls[:,0,...], flow)
        # l_dis = self.dis_loss(pred_flows, skls[:,0,...], flow)
        # l_con = self.con_loss(pred_flows, skls[:,0,...], flow)
        l_point = self.point_loss(masks, skls[:,0,...], flow)
        loss = self.hparams.sup_weight * l_sup + self.hparams.unsup_weight * l_unsup + \
                self.hparams.con_weight * l_con + self.hparams.point_weight * l_point
                # self.hparams.dis_weight * l_dis # + self.hparams.con_weight * l_con
        
        self.log(f'{mode}_loss', loss, sync_dist=True)
        self.log(f'{mode}_l_sup', l_sup, sync_dist=True)
        self.log(f'{mode}_l_unsup', l_unsup, sync_dist=True)
        # self.log(f'{mode}_l_dis', l_dis, sync_dist=True)
        self.log(f'{mode}_l_con', l_con, sync_dist=True)
        self.log(f'{mode}_l_point', l_point, sync_dist=True)

        return loss, mcons
        
    def _vis_flow(self, frms, pred_flows, masks, mcons, batch_idx, log=True, store=False):
        masks = F.sigmoid(masks)
        binarized_masks = (masks > 0.5).float()
        flows = pred_flows[-1]
        finals = flows * binarized_masks
        # for frm, flow, mask in zip(frms, pred_flow, masks):

        frm = frms[0]
        flow = flows[0]
        mask = masks[0]
        mcon = mcons[0]
        final = finals[0]

        frm0 = frm[0].detach().cpu().numpy().transpose(1, 2, 0)
        frm1 = frm[1].detach().cpu().numpy().transpose(1, 2, 0)
        flow = flow.detach().cpu().numpy().transpose(1, 2, 0)
        mask = mask.detach().cpu().numpy().transpose(1, 2, 0)
        mcon = mcon.detach().cpu().numpy().transpose(1, 2, 0)
        final = final.detach().cpu().numpy().transpose(1, 2, 0)

        frm0 = denormalize(frm0)
        frm1 = denormalize(frm1)
        flow = flow_to_image(flow)
        mask = denormalize(mask)
        mcon = denormalize(mcon)
        final = flow_to_image(final)
        
        if log:
            self.logger.log_image(key='frm0', images=[frm0])
            self.logger.log_image(key='frm1', images=[frm1])
            self.logger.log_image(key='flow', images=[flow])
            self.logger.log_image(key='mask', images=[mask])
            self.logger.log_image(key='mcon', images=[mcon])
            self.logger.log_image(key='final', images=[final])

        if store:
            fig = plt.figure(figsize=(20, 10))
            ax = fig.add_subplot(2, 3, 1)
            ax.imshow(frm0)
            ax = fig.add_subplot(2, 3, 2)
            ax.imshow(frm1)
            ax = fig.add_subplot(2, 3, 3)
            ax.imshow(flow)
            ax = fig.add_subplot(2, 3, 4)
            ax.imshow(mask)
            ax = fig.add_subplot(2, 3, 5)
            ax.imshow(mcon)
            ax = fig.add_subplot(2, 3, 6)
            ax.imshow(final)

            fig.savefig(f'{self.hparams.model_ckpt_dir}/vis/batch_{batch_idx}.png')
            plt.close(fig)
            plt.clf()

    def forward(self, x):
        if self.hparams.model_name == 'raft':
            return self.model(x, n_iters=self.hparams.n_iters)
        else:
            return self.model(x)
    
    def training_step(self, batch, batch_idx):
        frms, skls, flow = batch
        pred_flows, masks = self.forward(frms)
        loss, _ = self._calculate_loss(frms, skls, flow, pred_flows, masks, 'train')
        return loss
    
    def validation_step(self, batch, batch_idx):
        frms, skls, flow = batch
        pred_flows, masks = self.forward(frms)
        loss, mcons = self._calculate_loss(frms, skls, flow, pred_flows, masks, 'val')
        if batch_idx == 0:
            self._vis_flow(frms, pred_flows, masks, mcons, batch_idx, log=True, store=False)
        return loss
    
    def test_step(self, batch, batch_idx):
        frms, skls, flow = batch
        pred_flows, masks = self.forward(frms)
        loss, mcons = self._calculate_loss(frms, skls, flow, pred_flows, masks, 'test')
        self._vis_flow(frms, pred_flows, masks, mcons, batch_idx, log=False, store=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = create_optimizer(self.hparams, self.parameters())
        scheduler = create_scheduler(self.hparams, optimizer)
        return [optimizer], [scheduler]