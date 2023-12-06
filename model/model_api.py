from typing import Any
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import matplotlib.pyplot as plt
import os
from collections import OrderedDict

from model.seg.deeplabv3 import DeepLabV3
from model.seg.unet import UNet
from model.flow.raft import RAFT
from model.flow.gaflow import GAFlow
from loss.seg import BodySegLoss, JointRegLoss, JointSegLoss, FlowGuidedSegLoss, JointConsistLoss
from misc.vis import denormalize, flow_to_image, mask_to_image

def normalize_imagenet(img):
    return TF.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def create_model_seg(hparams):
    if hparams.seg_model_name == 'deeplabv3':
        return DeepLabV3(type=hparams.seg_type, pretrained=hparams.seg_pretrained, out_channels=hparams.seg_out_channels)
    elif hparams.seg_model_name == 'unet':
        return UNet(type=hparams.seg_type, pretrained=hparams.seg_pretrained, out_channels=hparams.seg_out_channels)
    
def create_model_flow(hparams):
    if hparams.flow_model_name == 'raft':
        model = RAFT(hparams)
    elif hparams.flow_model_name == 'gaflow':
        model = GAFlow(hparams)
    else:
        raise NotImplementedError
    
    sd = torch.load(hparams.flow_model_path, map_location='cpu')
    new_sd = OrderedDict()
    for k, v in sd.items():
        if k.startswith('module.'):
            new_sd[k[7:]] = v
        else:
            new_sd[k] = v
    model.load_state_dict(new_sd)

    return model

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
        self.model_seg = create_model_seg(hparams)
        self.model_flow = create_model_flow(hparams)
        
        self.body_seg_loss = BodySegLoss()
        self.joint_seg_loss = JointSegLoss()
        self.flow_guided_seg_loss = FlowGuidedSegLoss()
        self.joint_consist_loss = JointConsistLoss()

        os.makedirs(f'{self.hparams.model_ckpt_dir}/vis', exist_ok=True)
    
    def _calculate_loss(self, skls, flows, masks, feats, gt_masks, mode='train'):
        # skls: B 2 J 2
        # flows: B 2 H W
        # masks: B 2 J+2 H W
        # feats: B C H/8 W/8
        # gt_masks: B 4 H W
        # gt_dmaps: B 2 J H W

        gt_masks_body = gt_masks[:,:2,...]
        gt_masks_joint = gt_masks[:,2:,...]

        skls_ = skls.flatten(0, 1)
        masks_ = masks.flatten(0, 1)
        gt_masks_body_ = gt_masks_body.flatten(0, 1)
        gt_masks_joint_ = gt_masks_joint.flatten(0, 1)

        l_body = self.body_seg_loss(skls_, masks_[:,0], gt_masks_body_)
        l_joint = self.joint_seg_loss(skls_, masks_[:,2:], gt_masks_joint_)
        # l_joint = self.joint_seg_loss(F.sigmoid(masks_[:,2:])_)
        l_flow = self.flow_guided_seg_loss(masks[:,0,1,...], feats, flows)
        # l_consist = self.joint_consist_loss(masks[:,0,2:,...], masks[:,1,2:,...], skls[:,0,...], skls[:,1,...])
        l_consist = self.joint_consist_loss(masks[:,0,2:,...], masks[:,1,2:,...], flows)

        l_total = self.hparams.w_body * l_body + self.hparams.w_joint * l_joint + self.hparams.w_flow * l_flow + self.hparams.w_consist * l_consist
        
        self.log(f'{mode}_loss', l_total, sync_dist=True)
        self.log(f'{mode}_l_body', l_body, sync_dist=True)
        self.log(f'{mode}_l_joint', l_joint, sync_dist=True)
        self.log(f'{mode}_l_flow', l_flow, sync_dist=True)
        self.log(f'{mode}_l_consist', l_consist, sync_dist=True)

        return l_total
        
    def _vis_flow(self, frms, flows, masks, batch_idx, log=True, store=False):
        # frms: B 2 3 H W
        # flows: B 2 H W
        # masks: B 2 J+2 H W

        num_kps = masks.shape[2] - 2
        masks = F.sigmoid(masks)
        masks_body = (masks[:,:,0,...] > 0.5)
        masks_joint_neg = (torch.max(masks[:,:,2:,...], dim=2, keepdim=True)[0] < 0.5).float()
        masks_joint = torch.argmax(torch.cat([masks_joint_neg, masks[:,:,2:,...]], dim=2), dim=2) * masks_body
        masks_flow = (masks[:,0,1,...] > 0.5)

        for i, (frm, flow, mask_body, mask_joint, mask_flow) in enumerate(zip(frms, flows, masks_body, masks_joint, masks_flow)):
            frm0 = denormalize(frm[0].detach().cpu().numpy().transpose(1, 2, 0))
            frm1 = denormalize(frm[1].detach().cpu().numpy().transpose(1, 2, 0))
            flow = flow_to_image(flow.detach().cpu().numpy().transpose(1, 2, 0))
            mask_body0 = mask_to_image(mask_body[0].detach().cpu().numpy(), 1)
            mask_body1 = mask_to_image(mask_body[1].detach().cpu().numpy(), 1)
            mask_joint0 = mask_to_image(mask_joint[0].detach().cpu().numpy(), num_kps)
            mask_joint1 = mask_to_image(mask_joint[1].detach().cpu().numpy(), num_kps)
            mask_flow = mask_to_image(mask_flow.detach().cpu().numpy(), 1)

            if store:
                fig = plt.figure(figsize=(20, 10))
                ax = fig.add_subplot(2, 4, 1)
                ax.imshow(frm0)
                ax = fig.add_subplot(2, 4, 2)
                ax.imshow(frm1)
                ax = fig.add_subplot(2, 4, 3)
                ax.imshow(mask_body0)
                ax = fig.add_subplot(2, 4, 4)
                ax.imshow(mask_body1)
                ax = fig.add_subplot(2, 4, 5)
                ax.imshow(mask_joint0)
                ax = fig.add_subplot(2, 4, 6)
                ax.imshow(mask_joint1)
                ax = fig.add_subplot(2, 4, 7)
                ax.imshow(flow)
                ax = fig.add_subplot(2, 4, 8)
                ax.imshow(mask_flow)

                fig.savefig(f'{self.hparams.model_ckpt_dir}/vis/batch_{batch_idx}_{i}.png')
                plt.close(fig)
                plt.clf()

            if log and i == 0:
                self.logger.log_image(key='frm', images=[frm0,frm1])
                self.logger.log_image(key='mask_body', images=[mask_body0,mask_body1])
                self.logger.log_image(key='mask_joint', images=[mask_joint0,mask_joint1])
                self.logger.log_image(key='mask_flow', images=[mask_flow])
                self.logger.log_image(key='flow', images=[flow])

    def forward(self, x):
        frm0, frm1 = x[:,0,...], x[:,1,...]
        frm0_norm = normalize_imagenet(frm0)
        frm1_norm = normalize_imagenet(frm1)
        mask0, feat = self.model_seg(frm0_norm)
        mask1, _ = self.model_seg(frm1_norm)
        self.model_flow.eval()
        with torch.no_grad():
            flow = self.model_flow(frm0, frm1, test_mode=True)[-1]
        return flow, torch.stack([mask0, mask1], dim=1), feat
    
    def training_step(self, batch, batch_idx):
        frms, skls, _, gt_masks = batch
        flows, masks, feats = self.forward(frms)
        loss = self._calculate_loss(skls, flows, masks, feats, gt_masks, 'train')
        return loss
    
    def validation_step(self, batch, batch_idx):
        frms, skls, _, gt_masks = batch
        flows, masks, feats = self.forward(frms)
        loss = self._calculate_loss(skls, flows, masks, feats, gt_masks, 'val')
        if batch_idx == 0:
            self._vis_flow(frms, flows, masks, batch_idx, log=True, store=False)
        return loss
    
    def test_step(self, batch, batch_idx):
        frms, skls, _, gt_masks = batch
        flows, masks, feats = self.forward(frms)
        loss = self._calculate_loss(skls, flows, masks, feats, gt_masks, 'test')
        self._vis_flow(frms, flows, masks, batch_idx, log=False, store=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = create_optimizer(self.hparams, self.model_seg.parameters())
        scheduler = create_scheduler(self.hparams, optimizer)
        return [optimizer], [scheduler]