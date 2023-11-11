from typing import Any
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from model.flownet import FlowNetS
from model.raft import RAFT
from loss.flow import EPELoss, UnsupLoss, DisLoss, ConLoss

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
        self.sup_loss = EPELoss(hparams.flow_weights)
        self.unsup_loss = UnsupLoss(hparams.flow_weights)
        self.dis_loss = DisLoss(hparams.flow_weights)
        self.con_loss = ConLoss(hparams.flow_weights)
    
    def _calculate_loss(self, batch, mode='train'):
        frms, skls, flow = batch
        # if mode == 'train':
        mask = (flow != 0)
        pred_flows = self.forward(frms)
        loss = self.hparams.sup_weight * self.sup_loss(pred_flows, flow, mask) + \
                self.hparams.unsup_weight * self.unsup_loss(pred_flows, frms) + \
                self.hparams.dis_weight * self.dis_loss(pred_flows, skls, flow) + \
                self.hparams.con_weight * self.con_loss(pred_flows, skls, flow)
        
        self.log(f'{mode}_loss', loss)

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, 'train')
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, 'val')
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, 'test')
        return loss
    
    def configure_optimizers(self):
        optimizer = create_optimizer(self.hparams, self.parameters())
        scheduler = create_scheduler(self.hparams, optimizer)
        return [optimizer], [scheduler]