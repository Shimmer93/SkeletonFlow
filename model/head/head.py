import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import fvcore.nn.weight_init as weight_init
import math

import sys
sys.path.append('/home/zpengac/pose/SkeletonFlow')
from model.head.point import *
from misc.skeleton import get_min_max_from_skeletons

class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, activation=None):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                     groups, bias)
        self.activation = activation

    def forward(self, x):
        x = super(Conv2d, self).forward(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, C, ...) or (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
        classes (list): A list of length R that contains either predicted or ground truth class
            for eash predicted mask.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    if logits.shape[1] == 1:
        gt_class_logits = logits.clone()
    else:
        gt_class_logits = torch.topk(logits, k=2, dim=1)[0]
        gt_class_logits = gt_class_logits[:, 0:1, ...] - gt_class_logits[:, 1:2, ...]
    return -(torch.abs(gt_class_logits))

class ImplicitPointHead(nn.Module):
    """
    A point head multi-layer perceptron which we model with conv1d layers with kernel 1. The head
    takes both fine-grained features and instance-wise MLP parameters as its input.
    """

    def __init__(self, cfg, in_channels, num_classes=1):
        """
        The following attributes are parsed from config:
            channels: the output dimension of each FC layers
            num_layers: the number of FC layers (including the final prediction layer)
            image_feature_enabled: if True, fine-grained image-level features are used
            positional_encoding_enabled: if True, positional encoding is used
        """
        super(ImplicitPointHead, self).__init__()
        # fmt: off
        self.num_layers                         = cfg.point_head_num_fc + 1
        self.channels                           = cfg.point_head_fc_dim
        self.image_feature_enabled              = cfg.image_feature_enabled
        self.positional_encoding_enabled        = cfg.pos_enc_enabled
        self.num_classes                        = num_classes
        self.in_channels                        = in_channels
        # fmt: on

        if not self.image_feature_enabled:
            self.in_channels = 0
        if self.positional_encoding_enabled:
            self.in_channels += self.channels
            self.register_buffer("positional_encoding_gaussian_matrix", torch.randn((2, self.channels//2)))

        assert self.in_channels > 0

        mlp_layers = []
        for l in range(self.num_layers):
            if l == 0:
                mlp_layers.append(nn.Linear(self.in_channels, self.channels))
                mlp_layers.append(nn.ReLU())
                mlp_layers.append(nn.Dropout(0.1))
            elif l == self.num_layers - 1:
                mlp_layers.append(nn.Linear(self.channels, self.num_classes))
            else:
                mlp_layers.append(nn.Linear(self.channels, self.channels))
                mlp_layers.append(nn.ReLU())
                mlp_layers.append(nn.Dropout(0.1))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, fine_grained_features, point_coords):
        # features: [R, channels, K]
        # point_coords: [R, K, 2]
        num_instances = fine_grained_features.size(0)
        num_points = fine_grained_features.size(2)

        if num_instances == 0:
            return torch.zeros((0, 1, num_points), device=fine_grained_features.device)

        if self.positional_encoding_enabled:
            # locations: [R*K, 2]
            locations = 2 * point_coords.reshape(num_instances * num_points, 2) - 1
            locations = locations @ self.positional_encoding_gaussian_matrix.to(locations.device)
            locations = 2 * np.pi * locations
            locations = torch.cat([torch.sin(locations), torch.cos(locations)], dim=1)
            # locations: [R, C, K]
            locations = locations.reshape(num_instances, num_points, self.channels).permute(0, 2, 1)
            if not self.image_feature_enabled:
                fine_grained_features = locations
            else:
                fine_grained_features = torch.cat([locations, fine_grained_features], dim=1)

        # features [R, C, K]
        mask_feat = fine_grained_features.reshape(num_instances, self.in_channels, num_points)

        B, C, L = mask_feat.shape
        mask_feat = mask_feat.transpose(1, 2).reshape(B*L, C)
        point_logits = self.mlp(mask_feat)
        point_logits = point_logits.reshape(B, L, self.num_classes).transpose(1, 2)

        return point_logits

class ImplicitPointRendMaskHead(nn.Module):
    def __init__(self, cfg, in_channels, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        # fmt: off
        self.mask_point_on = True  # always on
        self.mask_point_train_num_points        = cfg.train_num_points
        # next two parameters are use in the adaptive subdivions inference procedure
        self.mask_point_subdivision_steps       = cfg.subdivision_steps
        self.mask_point_subdivision_num_points  = cfg.subdivision_num_points
        # fmt: on
        self.scale = cfg.scale #1.0 / 4
        self.feat_scale = int((2 ** self.mask_point_subdivision_steps) * self.scale)

        # inference parameters
        self.mask_point_subdivision_init_resolution = int(
            math.sqrt(self.mask_point_subdivision_num_points)
        )
        assert (
            self.mask_point_subdivision_init_resolution
            * self.mask_point_subdivision_init_resolution
            == self.mask_point_subdivision_num_points
        )

        self.point_head = ImplicitPointHead(cfg, in_channels, self.num_classes)

    def forward(self, features):
        """
        Args:
            features: B C H W
        """
        pass

    def _get_point_logits(self, fine_grained_features, point_coords):
        return self.point_head(fine_grained_features, point_coords)
    
    def _point_pooler(self, features, point_coords):
        # sample image-level features
        point_fine_grained_features, _ = point_sample_fine_grained_features(
            features, self.scale, point_coords
        )
        return point_fine_grained_features

    def _subdivision_inference(self, features):
        assert not self.training

        mask_point_subdivision_num_points = features.shape[-2] * features.shape[-1] // (self.feat_scale ** 2)
        mask_logits = None
        # +1 here to include an initial step to generate the coarsest mask
        # prediction with init_resolution, when mask_logits is None.
        # We compute initial mask by sampling on a regular grid. coarse_mask
        # can be used as initial mask as well, but it's typically very low-res
        # so it will be completely overwritten during subdivision anyway.
        for _ in range(self.mask_point_subdivision_steps + 1):
            if mask_logits is None:
                # print(features.shape[-2]//self.feat_scale, features.shape[-1]//self.feat_scale)
                point_coords = generate_regular_grid_point_coords(
                    features.shape[0],
                    features.shape[-2]//self.feat_scale,
                    features.shape[-1]//self.feat_scale,
                    features.device,
                )
            else:
                mask_logits = F.interpolate(
                    mask_logits, scale_factor=2, mode="bilinear", align_corners=False
                )
                uncertainty_map = calculate_uncertainty(mask_logits)
                point_indices, point_coords = get_uncertain_point_coords_on_grid(
                    uncertainty_map, mask_point_subdivision_num_points
                )

            # Run the point head for every point in point_coords
            fine_grained_features = self._point_pooler(features, point_coords)
            point_logits = self._get_point_logits(
                fine_grained_features, point_coords
            )

            if mask_logits is None:
                # Create initial mask_logits using point_logits on this regular grid
                R, C, _ = point_logits.shape
                mask_logits = point_logits.reshape(
                    R,
                    C,
                    features.shape[-2]//self.feat_scale,
                    features.shape[-1]//self.feat_scale,
                )
            else:
                # Put point predictions to the right places on the upsampled grid.
                R, C, H, W = mask_logits.shape
                point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
                mask_logits = (
                    mask_logits.reshape(R, C, H * W)
                    .scatter_(2, point_indices, point_logits)
                    .view(R, C, H, W)
                )
        return mask_logits

class BodyMaskHead(ImplicitPointRendMaskHead):
    def __init__(self, cfg, in_channels, num_classes=1):
        self.flag = True
        self.num_classes = num_classes
        super().__init__(cfg, in_channels)

    def forward(self, features, skls=None, gt_masks=None):
        """
        Args:
            features: B C H W
        """
        if self.training:
            point_coords, point_labels = self._sample_train_points_with_skeleton(features, skls, gt_masks)
            point_fine_grained_features = self._point_pooler(features, point_coords)
            point_logits = self._get_point_logits(
                point_fine_grained_features, point_coords
            )

            return point_logits, point_labels
        else:
            return self._subdivision_inference(features)
    
    def _sample_train_points_with_skeleton(self, features, skls, gt_masks):
        assert self.training

        hf = features.shape[-2]
        wf = features.shape[-1]

        h = int(hf // self.scale)
        w = int(wf // self.scale)

        y_mins, y_maxs, x_mins, x_maxs = get_min_max_from_skeletons(skls, h, w)

        point_coords = []
        point_labels = []
        for(i, (y_min, y_max, x_min, x_max)) in enumerate(zip(y_mins, y_maxs, x_mins, x_maxs)):
            num_samples = self.mask_point_train_num_points // 2
            neg_mask = torch.ones((h, w), dtype=torch.bool, device=features.device)
            neg_mask[y_min:y_max, x_min:x_max] = False
            if torch.sum(neg_mask) == 0:
                neg_mask[0, 0] = True
            neg_idxs = torch.nonzero(neg_mask).flip(1)
            while len(neg_idxs) < num_samples:
                neg_idxs = torch.cat([neg_idxs, neg_idxs], dim=0)
            neg_idxs = neg_idxs[torch.randperm(len(neg_idxs)),:][:num_samples]
            pos_mask = (gt_masks[i] > 0)
            pos_idxs = torch.nonzero(pos_mask).flip(1)
            while len(pos_idxs) < num_samples:
                pos_idxs = torch.cat([pos_idxs, pos_idxs], dim=0)
            pos_idxs = pos_idxs[torch.randperm(len(pos_idxs)),:][:num_samples]
            point_coords.append(torch.cat([neg_idxs, pos_idxs], dim=0))
            point_labels.append(torch.cat([torch.zeros(num_samples), torch.ones(num_samples)], dim=0))

        point_coords = torch.stack(point_coords, dim=0).to(features.device)
        point_coords = point_coords / torch.tensor([w-1, h-1], dtype=torch.float, device=features.device).unsqueeze(0)
        point_labels = torch.stack(point_labels, dim=0).to(features.device)

        return point_coords, point_labels
    
class JointMaskHead(ImplicitPointRendMaskHead):
    def __init__(self, cfg, in_channels, num_classes=1):
        super().__init__(cfg, in_channels, num_classes)

    def forward(self, features, skls=None, gt_masks=None):
        """
        Args:
            features: B C H W
        """
        if self.training:
            point_coords, point_labels = self._sample_train_points_with_skeleton(features, skls, gt_masks)
            point_fine_grained_features = self._point_pooler(features, point_coords)
            point_logits = self._get_point_logits(
                point_fine_grained_features, point_coords
            )

            return point_logits, point_labels
        else:
            return self._subdivision_inference(features)

    def _sample_train_points_with_skeleton(self, features, skls, gt_masks):
        assert self.training

        hf = features.shape[-2]
        wf = features.shape[-1]

        h = int(hf // self.scale)
        w = int(wf // self.scale)

        y_mins, y_maxs, x_mins, x_maxs = get_min_max_from_skeletons(skls, h, w)

        point_coords = []
        point_labels = []
        for(i, (y_min, y_max, x_min, x_max)) in enumerate(zip(y_mins, y_maxs, x_mins, x_maxs)):
            coords_i = []
            labels_i = []

            num_samples_neg = self.mask_point_train_num_points // 2
            # neg_mask = (gt_masks[i].sum(dim=0) == 0)
            neg_mask = torch.ones((h, w), dtype=torch.bool, device=features.device)
            neg_mask[y_min:y_max, x_min:x_max] = False
            if torch.sum(neg_mask) == 0:
                neg_mask[0, 0] = True
            neg_idxs = torch.nonzero(neg_mask).flip(1)

            neg_idxs = neg_idxs[torch.randperm(len(neg_idxs)),:][:num_samples_neg]
            while len(neg_idxs) < num_samples_neg:
                neg_idxs = torch.cat([neg_idxs, neg_idxs], dim=0)
            coords_i.append(neg_idxs)
            labels_i.append(torch.zeros(num_samples_neg))

            num_samples_pos = self.mask_point_train_num_points // 8
            for j in range(gt_masks.shape[1]):
                pos_mask = (gt_masks[i][j] > 0)
                pos_idxs = torch.nonzero(pos_mask).flip(1)
                while len(pos_idxs) < num_samples_pos:
                    pos_idxs = torch.cat([pos_idxs, pos_idxs], dim=0)
                pos_idxs = pos_idxs[torch.randperm(len(pos_idxs)),:][:num_samples_pos]
                coords_i.append(pos_idxs)
                labels_i.append(torch.ones(num_samples_pos) * (j+1))

            point_coords.append(torch.cat(coords_i, dim=0))
            point_labels.append(torch.cat(labels_i, dim=0))

        point_coords = torch.stack(point_coords, dim=0).to(features.device)
        point_coords = point_coords / torch.tensor([w-1, h-1], dtype=torch.float, device=features.device).unsqueeze(0)
        point_labels = torch.stack(point_labels, dim=0).to(features.device)

        return point_coords, point_labels
    
class FlowMaskHead(ImplicitPointRendMaskHead):
    def __init__(self, cfg, in_channels, num_classes=1):
        super().__init__(cfg, in_channels, num_classes)

    def forward(self, features, flows=None):
        """
        Args:
            features: B C H W
        """
        if self.training:
            # parameters = self.parameter_head(self._roi_pooler(features))

            point_coords, point_labels = self._sample_train_points_with_flow(features, flows)
            point_fine_grained_features = self._point_pooler(features, point_coords)
            point_logits = self._get_point_logits(
                point_fine_grained_features, point_coords
            )

            return point_logits, point_labels
        else:
            # parameters = self.parameter_head(self._roi_pooler(features))
            return self._subdivision_inference(features)

    def _sample_train_points_with_flow(self, features, flows):
        assert self.training

        hf = features.shape[-2]
        wf = features.shape[-1]

        h = int(hf // self.scale)
        w = int(wf // self.scale)

        # with torch.no_grad():
        #     flows0 = flows.detach().clone()
        #     flows_norm = torch.norm(flows0, dim=1, keepdim=True)
        #     flows_norm_max = flows_norm.flatten(2).max(dim=2)[0].unsqueeze(1).unsqueeze(1)

        #     flows1 = flows.detach().clone()
        #     flows_mean = torch.mean(flows1, dim=(2, 3), keepdim=True)
        #     flows_centered = flows1 - flows_mean
        #     flows_centered_norm = torch.norm(flows_centered, dim=1, keepdim=True)
        #     flows_centered_norm_max = flows_centered_norm.flatten(2).max(dim=2)[0].unsqueeze(1).unsqueeze(1)
            
        #     pos_masks1 = (flows_norm > flows_norm_max * 0.8).squeeze(1)
        #     neg_masks1 = (flows_norm < flows_norm_max * 0.2).squeeze(1)
        #     pos_masks2 = (flows_centered_norm > flows_centered_norm_max * 0.8).squeeze(1)
        #     neg_masks2 = (flows_centered_norm < flows_centered_norm_max * 0.2).squeeze(1)
        #     pos_masks = pos_masks2 & ~neg_masks1
        #     neg_masks = neg_masks2 & ~pos_masks1

        #     for i in range(len(pos_masks)):
        #         print(torch.sum(pos_masks[i]), torch.sum(neg_masks[i]))

        flows_mean = torch.mean(flows, dim=(2, 3), keepdim=True)
        flows_centered = flows - flows_mean
        flows_centered_norm = torch.norm(flows_centered, dim=1, keepdim=True)
        flows_centered_norm_max = flows_centered_norm.flatten(2).max(dim=2)[0].unsqueeze(1).unsqueeze(1)
        
        pos_masks = (flows_centered_norm > flows_centered_norm_max * 0.8).squeeze(1)
        neg_masks = (flows_centered_norm < flows_centered_norm_max * 0.2).squeeze(1)

        point_coords = []
        point_labels = []

        for(i, (pos_mask, neg_mask)) in enumerate(zip(pos_masks, neg_masks)):
            num_samples = self.mask_point_train_num_points // 2
            neg_idxs = torch.nonzero(neg_mask).flip(1)
            while len(neg_idxs) < num_samples:
                neg_idxs = torch.cat([neg_idxs, neg_idxs], dim=0)
            neg_idxs = neg_idxs[torch.randperm(len(neg_idxs)),:][:num_samples]
            pos_idxs = torch.nonzero(pos_mask).flip(1)
            while len(pos_idxs) < num_samples:
                pos_idxs = torch.cat([pos_idxs, pos_idxs], dim=0)
            pos_idxs = pos_idxs[torch.randperm(len(pos_idxs)),:][:num_samples]
            point_coords.append(torch.cat([neg_idxs, pos_idxs], dim=0))
            point_labels.append(torch.cat([torch.zeros(num_samples), torch.ones(num_samples)], dim=0))

        point_coords = torch.stack(point_coords, dim=0).to(features.device)
        point_coords = point_coords / torch.tensor([w-1, h-1], dtype=torch.float, device=features.device).unsqueeze(0)
        point_labels = torch.stack(point_labels, dim=0).to(features.device)

        return point_coords, point_labels
    
if __name__ == '__main__':
    class cfg:
        point_head_num_fc = 3
        point_head_fc_dim = 128
        train_num_points = 196
        subdivision_steps = 3
        subdivision_num_points = 784
        image_feature_enabled = True
        pos_enc_enabled = True
        scale = 1.0 / 4

    in_channels = 256

    head = BodyMaskHead(cfg, in_channels)

    features = torch.rand(2, 256, 56, 56)
    skls = (torch.rand(2, 17, 2) * 224).type(torch.int)
    gt_masks = torch.rand(2, 224, 224) > 0.5

    point_logits, point_labels = head(features, skls, gt_masks)
    print(point_logits.shape)
    print(point_labels.shape)
    print(point_logits.max(), point_logits.min())

    features = torch.rand(2, 256, 60, 80)

    head.eval()
    mask_logits = head(features)
    print(mask_logits.shape)
    print(mask_logits.max(), mask_logits.min())