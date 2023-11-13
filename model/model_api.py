from typing import Any
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from model.flownet import FlowNetS
from model.raft import RAFT
from loss.flow import EPELoss, UnsupLoss, DisLoss, ConLoss
from misc.vis import denormalize, flow_to_image

def create_model(hparams):
    if hparams.model_name == 'flownet':
        return FlowNetS()
    elif hparams.model_name == 'raft':
        return RAFT(hparams)
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
        self.dis_loss = DisLoss(hparams.gamma)
        # self.con_loss = ConLoss(hparams.gamma)
    
    def _calculate_loss(self, frms, skls, flow, pred_flows, mode='train'):
        flow_mag = torch.sum(flow**2, dim=1, keepdim=True)
        mask = (flow_mag != 0)
        l_sup = self.sup_loss(pred_flows, flow, mask)
        l_unsup = self.unsup_loss(pred_flows, frms)
        l_dis = self.dis_loss(pred_flows, skls[:,0,...], flow)
        # l_con = self.con_loss(pred_flows, skls[:,0,...], flow)
        loss = self.hparams.sup_weight * l_sup + self.hparams.unsup_weight * l_unsup + \
                self.hparams.dis_weight * l_dis # + self.hparams.con_weight * l_con
        
        self.log(f'{mode}_loss', loss, sync_dist=True)
        self.log(f'{mode}_l_sup', l_sup, sync_dist=True)
        self.log(f'{mode}_l_unsup', l_unsup, sync_dist=True)
        self.log(f'{mode}_l_dis', l_dis, sync_dist=True)
        # self.log(f'{mode}_l_con', l_con, sync_dist=True)

        return loss
        
    def _vis_flow(self, frms, pred_flows):
        pred_flow = pred_flows[-1]
        for frm, flow in zip(frms, pred_flow):
            frm0 = frm[0].detach().cpu().numpy().transpose(1, 2, 0)
            frm1 = frm[1].detach().cpu().numpy().transpose(1, 2, 0)
            flow = flow.detach().cpu().numpy().transpose(1, 2, 0)
            frm0 = denormalize(frm0)
            frm1 = denormalize(frm1)
            flow = flow_to_image(flow)
            self.logger.log_image(key='frm0', images=[frm0])
            self.logger.log_image(key='frm1', images=[frm1])
            self.logger.log_image(key='flow', images=[flow])

    def forward(self, x):
        if self.hparams.model_name == 'raft':
            return self.model(x, n_iters=self.hparams.n_iters)
        else:
            return self.model(x)
    
    def training_step(self, batch, batch_idx):
        frms, skls, flow = batch
        pred_flows = self.forward(frms)
        loss = self._calculate_loss(frms, skls, flow, pred_flows, 'train')
        return loss
    
    def validation_step(self, batch, batch_idx):
        frms, skls, flow = batch
        pred_flows = self.forward(frms)
        loss = self._calculate_loss(frms, skls, flow, pred_flows, 'val')
        if batch_idx == 0:
            self._vis_flow(frms, pred_flows)
        return loss
    
    def test_step(self, batch, batch_idx):
        frms, skls, flow = batch
        pred_flows = self.forward(frms)
        loss = self._calculate_loss(frms, skls, flow, pred_flows, 'test')
        if batch_idx == 0:
            self._vis_flow(frms, pred_flows)
        return loss
    
    def configure_optimizers(self):
        optimizer = create_optimizer(self.hparams, self.parameters())
        scheduler = create_scheduler(self.hparams, optimizer)
        return [optimizer], [scheduler]