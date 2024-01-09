import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import matplotlib.pyplot as plt
import os
from collections import OrderedDict

import matplotlib.pyplot as plt

from model.seg.deeplabv3 import DeepLabV3
from model.seg.unet import UNet, UNetFlow

from model.flow.raft import RAFT

from model.head import BodyMaskHead, JointMaskHead, FlowMaskHead
# from model.flow.gaflow import GAFlow
from loss.loss import JointSegTrainLoss, JointConsistLoss, BodySegValLoss, JointSegValLoss, FlowSegValLoss
from misc.vis import denormalize, flow_to_image, mask_to_image, mask_to_joint_images
from misc.utils import write_psm

def normalize_imagenet(img):
    return TF.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def create_model_seg(hparams):
    if hparams.seg_model_name == 'deeplabv3':
        return DeepLabV3(type=hparams.seg_type, pretrained=hparams.seg_pretrained, num_joints=hparams.num_joints)
    elif hparams.seg_model_name == 'unet':
        return UNet(type=hparams.seg_type, pretrained=hparams.seg_pretrained, num_joints=hparams.num_joints), \
                UNetFlow(type=hparams.seg_type)
    else:
        raise NotImplementedError
    
def create_model_flow(hparams):
    if hparams.flow_model_name == 'raft':
        model = RAFT(hparams)
    # elif hparams.flow_model_name == 'gaflow':
    #     model = GAFlow(hparams)
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
        self.model_seg, self.model_seg_flow = create_model_seg(hparams)
        self.model_flow = create_model_flow(hparams)

        self.head_joint = JointMaskHead(hparams, input_shape=[256, 14, 14], num_classes=hparams.num_joints)
        self.head_body = BodyMaskHead(hparams, input_shape=[256, 14, 14])
        self.head_flow = FlowMaskHead(hparams, input_shape=[256, 14, 14])
        
        self.body_seg_loss = nn.BCEWithLogitsLoss()
        self.joint_seg_loss = JointSegTrainLoss()
        self.flow_guided_seg_loss = nn.BCEWithLogitsLoss()
        # self.joint_consist_loss = JointConsistLoss()

        self.body_seg_metric = BodySegValLoss()
        self.joint_seg_metric = JointSegValLoss()
        self.flow_seg_metric = FlowSegValLoss()

        os.makedirs(f'{self.hparams.model_ckpt_dir}/vis', exist_ok=True)
    
    def _calculate_loss(self, input, output, mode='train'):
        # skls: B 2 J 2
        # flows: B 2 H W
        # masks: B 2 J+2 H W
        # feats: B C H/8 W/8
        # gt_masks: B 2+2J H W
        # gt_dmaps: B 2 J H W

        if mode == 'train':
            l_body = self.body_seg_loss(output['point_logits_body'], output['point_labels_body'])
            l_joint = self.joint_seg_loss(output['point_logits_joint'], output['point_labels_joint'])
            l_flow = self.flow_guided_seg_loss(output['point_logits_flow'], output['point_labels_flow'])
            l_total = self.hparams.w_body * l_body + self.hparams.w_joint * l_joint + self.hparams.w_flow * l_flow
            self.log(f'{mode}_l_body', l_body, sync_dist=True)
            self.log(f'{mode}_l_joint', l_joint, sync_dist=True)
            self.log(f'{mode}_l_flow', l_flow, sync_dist=True)
            # self.log(f'{mode}_l_consist', l_consist, sync_dist=True)
            self.log(f'{mode}_loss', l_total, sync_dist=True)
            return l_total
        
        else:
            skl_ = input['skls'].flatten(0, 1)
            gt_mask_body_ = input['masks'][:,:2,...].flatten(0, 1)
            gt_masks_joint_ = torch.chunk(input['masks'][:,2:,...], 2, dim=1)
            gt_masks_joint_ = torch.stack([gt_masks_joint_[0], gt_masks_joint_[1]], dim=1)
            gt_masks_joint_ = gt_masks_joint_.flatten(0, 1)
            
            m_body = self.body_seg_metric(skl_, output['mask_body'], gt_mask_body_)
            m_joint = self.joint_seg_metric(skl_, output['masks_joint'], gt_masks_joint_)
            m_flow = self.flow_seg_metric(output['mask_flow'], output['flow'])
            m_total = self.hparams.w_body * m_body + self.hparams.w_joint * m_joint + self.hparams.w_flow * m_flow
            self.log(f'{mode}_m_body', m_body, sync_dist=True)
            self.log(f'{mode}_m_joint', m_joint, sync_dist=True)
            self.log(f'{mode}_m_flow', m_flow, sync_dist=True)
            self.log(f'{mode}_metric', m_total, sync_dist=True)
            return m_total

        
    def _vis_flow(self, input, output, batch_idx, log=True, store=False):
        # frms: B 2 3 H W
        # flows: B 2 H W
        # masks: B 2 J+2 H W

        frms = input['frms']
        flows = output['flow']
        masks_joints = output['masks_joint']
        masks_body = output['mask_body']
        masks_flow = output['mask_flow']

        num_kps = self.hparams.num_joints
        masks_body = (F.sigmoid(masks_body) > 0.5).float()
        masks_joints = (F.sigmoid(masks_joints) > 0.5).float()
        masks_joints_neg = (torch.max(masks_joints, dim=1, keepdim=True)[0] < 0.5).float()
        masks_joint = torch.argmax(torch.cat([masks_joints_neg, masks_joints], dim=1), dim=1) # * masks_body
        masks_flow = (F.sigmoid(masks_flow) > 0.5).float()

        for i, (frm, flow, mask_body, mask_joint, mask_flow, mask_joints) in enumerate(zip(frms, flows, masks_body, masks_joint, masks_flow, masks_joints)):
            frm0 = frm[0].detach().cpu().numpy().transpose(1, 2, 0)
            frm1 = frm[1].detach().cpu().numpy().transpose(1, 2, 0)
            flow = flow_to_image(flow.detach().cpu().numpy().transpose(1, 2, 0))
            mask_body = mask_to_image(mask_body.detach().cpu().numpy(), 1)
            mask_joint = mask_to_image(mask_joint.detach().cpu().numpy(), num_kps)
            mask_flow = mask_to_image(mask_flow.detach().cpu().numpy(), 1)
            mask_joints = mask_to_joint_images(mask_joints.detach().cpu().numpy(), num_kps)

            if store:
                data = [frm0, frm1, mask_body, mask_joint, flow, mask_flow]

                fig = plt.figure(figsize=(15, 8))
                for j, d in enumerate(data):
                    ax = fig.add_subplot(2, 3, j+1)
                    ax.imshow(d)

                fig.savefig(f'{self.hparams.model_ckpt_dir}/vis/batch_{batch_idx}_{i}.png')
                plt.close(fig)
                plt.clf()

                fig = plt.figure(figsize=(10, 5))
                for k in range(num_kps):
                    ax = fig.add_subplot(3, 5, k+1)
                    ax.imshow(mask_joints[k])
                fig.savefig(f'{self.hparams.model_ckpt_dir}/vis/batch_{batch_idx}_{i}_joints.png')
                plt.close(fig)
                plt.clf()

            if log and i == 0:
                self.logger.log_image(key='frm', images=[frm0,frm1])
                self.logger.log_image(key='mask_body', images=[mask_body])
                self.logger.log_image(key='mask_joint', images=[mask_joint])
                self.logger.log_image(key='mask_flow', images=[mask_flow])
                self.logger.log_image(key='flow', images=[flow])

    def _save_psm(self, input, output):
        B = input['frms'].shape[0]
        for i in range(B):
            frm0_fn = input['frm0_fn'][i]
            psm_fn = frm0_fn.replace('Rename_Images', 'PSM_v1')#.replace('.png', '.jpg')#.replace('.jpg', '.psm')
            os.makedirs(os.path.dirname(psm_fn), exist_ok=True)
            mask_joints = F.sigmoid(output['masks_joint'][i]) * (F.sigmoid(output['mask_body'][i]).unsqueeze(0) > 0.5).float()
            mask_body = F.sigmoid(output['mask_body'][i])
            write_psm(psm_fn, mask_joints, body_mask=mask_body, rescale_ratio=4.0)

    def forward(self, x):
        frm0, frm1 = x['frms'][:,0,...], x['frms'][:,1,...]
        skl_ = x['skls'].flatten(0, 1)
        gt_mask_body_ = x['masks'][:,:2,...].flatten(0, 1)
        gt_masks_joint_ = torch.chunk(x['masks'][:,2:,...], 2, dim=1)
        gt_masks_joint_ = torch.stack([gt_masks_joint_[0], gt_masks_joint_[1]], dim=1)
        gt_masks_joint_ = gt_masks_joint_.flatten(0, 1)

        frm_ = x['frms'].flatten(0, 1)
        frm_norm_ = normalize_imagenet(frm_)
        feat_body_, feat_joint_ = self.model_seg(frm_norm_)
        with torch.no_grad():
            flows = self.model_flow(frm0, frm1, test_mode=True)
        feat_flow = self.model_seg_flow(flows[-1].clone().detach())

        ret_body = self.head_body(feat_body_, skl_, gt_mask_body_)
        ret_joint = self.head_joint(feat_joint_, skl_, gt_masks_joint_)
        ret_flow = self.head_flow(feat_flow, flows[-1])

        if self.training:
            return {
                'flow': flows[-1].detach(),
                'point_logits_joint': ret_joint[0],
                'point_labels_joint': ret_joint[1],
                'point_logits_body': ret_body[0].squeeze(1),
                'point_labels_body': ret_body[1],
                'point_logits_flow': ret_flow[0].squeeze(1),
                'point_labels_flow': ret_flow[1]
            }
        else:
            return {
                'flow': flows[-1].detach(),
                'masks_joint': ret_joint,
                'mask_body': ret_body.squeeze(1),
                'mask_flow': ret_flow.squeeze(1)
            }
    
    def training_step(self, batch, batch_idx):
        input = batch
        output = self.forward(batch)
        loss = self._calculate_loss(input, output, 'train')
        return loss
    
    def validation_step(self, batch, batch_idx):
        input = batch
        output = self.forward(batch)
        loss = self._calculate_loss(input, output, 'val')
        if batch_idx == 0:
            self._vis_flow(input, output, batch_idx, log=True, store=False)
        return loss
    
    def test_step(self, batch, batch_idx):
        input = batch
        output = self.forward(batch)
        loss = self._calculate_loss(input, output, 'test')
        self._vis_flow(input, output, batch_idx, log=False, store=True)
        # self._save_psm(input, output)
        return loss
    
    def configure_optimizers(self):
        optimizer = create_optimizer(self.hparams, list(self.model_seg.parameters())+list(self.head_body.parameters())+list(self.head_joint.parameters())+list(self.head_flow.parameters())+list(self.model_seg_flow.parameters()))
        scheduler = create_scheduler(self.hparams, optimizer)
        return [optimizer], [scheduler]