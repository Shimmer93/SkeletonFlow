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

import matplotlib.pyplot as plt

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
    # if logits.shape[1] == 1:
    #     gt_class_logits = logits.clone()
    # else:
    #     gt_class_logits = logits[
    #         torch.arange(logits.shape[0], device=logits.device), 1
    #     ].unsqueeze(1)
    # return -(torch.abs(gt_class_logits))

class ImplicitPointHead(nn.Module):
    """
    A point head multi-layer perceptron which we model with conv1d layers with kernel 1. The head
    takes both fine-grained features and instance-wise MLP parameters as its input.
    """

    def __init__(self, cfg, input_shape, num_classes=1):
        """
        The following attributes are parsed from config:
            channels: the output dimension of each FC layers
            num_layers: the number of FC layers (including the final prediction layer)
            image_feature_enabled: if True, fine-grained image-level features are used
            positional_encoding_enabled: if True, positional encoding is used
        """
        super(ImplicitPointHead, self).__init__()
        # fmt: off
        self.num_layers                         = cfg.POINT_HEAD_NUM_FC + 1
        self.channels                           = cfg.POINT_HEAD_FC_DIM
        self.image_feature_enabled              = cfg.IMPLICIT_POINTREND_IMAGE_FEATURE_ENABLED
        self.positional_encoding_enabled        = cfg.IMPLICIT_POINTREND_POS_ENC_ENABLED
        self.num_classes                        = num_classes
        self.in_channels                        = input_shape[0]
        # fmt: on

        if not self.image_feature_enabled:
            self.in_channels = 0
        if self.positional_encoding_enabled:
            self.in_channels += 256
            self.register_buffer("positional_encoding_gaussian_matrix", torch.randn((2, 128)))

        assert self.in_channels > 0

        # num_weight_params, num_bias_params = [], []
        # assert self.num_layers >= 2
        # for l in range(self.num_layers):
        #     if l == 0:
        #         # input layer
        #         num_weight_params.append(self.in_channels * self.channels)
        #         # print(f'self.in_channels: {self.in_channels}, self.channels: {self.channels}, product: {self.in_channels * self.channels}')
        #         num_bias_params.append(self.channels)
        #     elif l == self.num_layers - 1:
        #         # output layer
        #         num_weight_params.append(self.channels * self.num_classes)
        #         # print(f'self.channels: {self.channels}, self.num_classes: {self.num_classes}, product: {self.channels * self.num_classes}')
        #         num_bias_params.append(self.num_classes)
        #     else:
        #         # intermediate layer
        #         num_weight_params.append(self.channels * self.channels)
        #         # print(f'self.channels: {self.channels}, self.channels: {self.channels}, product: {self.channels * self.channels}')
        #         num_bias_params.append(self.channels)

        # self.num_weight_params = num_weight_params
        # self.num_bias_params = num_bias_params
        # self.num_params = sum(num_weight_params) + sum(num_bias_params)

        # self.params = nn.Parameter(torch.randn((1, self.num_params), requires_grad=True))
        mlp_layers = []
        for l in range(self.num_layers):
            if l == 0:
                mlp_layers.append(nn.Linear(self.in_channels, self.channels))
                # mlp_layers.append(nn.BatchNorm1d(self.channels))
                mlp_layers.append(nn.ReLU())
                # mlp_layers.append(nn.Dropout(0.1))
            elif l == self.num_layers - 1:
                mlp_layers.append(nn.Linear(self.channels, self.num_classes))
            else:
                mlp_layers.append(nn.Linear(self.channels, self.channels))
                # mlp_layers.append(nn.BatchNorm1d(self.channels))
                mlp_layers.append(nn.ReLU())
                # mlp_layers.append(nn.Dropout(0.1))
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
            locations = locations.reshape(num_instances, num_points, 256).permute(0, 2, 1)
            if not self.image_feature_enabled:
                fine_grained_features = locations
            else:
                fine_grained_features = torch.cat([locations, fine_grained_features], dim=1)

        # features [R, C, K]
        mask_feat = fine_grained_features.reshape(num_instances, self.in_channels, num_points)

        # weights, biases = self._parse_params(
        #     self.params.repeat(num_instances, 1),
        #     self.in_channels,
        #     self.channels,
        #     self.num_classes,
        #     self.num_weight_params,
        #     self.num_bias_params,
        # )

        # point_logits = self._dynamic_mlp(mask_feat, weights, biases, num_instances)
        # B, C, L = mask_feat.shape
        point_logits = self.mlp(mask_feat.transpose(1, 2)).transpose(1, 2)
        point_logits = point_logits.reshape(-1, self.num_classes, num_points)

        return point_logits

    @staticmethod
    def _dynamic_mlp(features, weights, biases, num_instances):
        assert features.dim() == 3, features.dim()
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = torch.einsum("nck,ndc->ndk", x, w) + b
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    @staticmethod
    def _parse_params(
        pred_params,
        in_channels,
        channels,
        num_classes,
        num_weight_params,
        num_bias_params,
    ):
        assert pred_params.dim() == 2
        assert len(num_weight_params) == len(num_bias_params)
        assert pred_params.size(1) == sum(num_weight_params) + sum(num_bias_params)

        num_instances = pred_params.size(0)
        num_layers = len(num_weight_params)

        params_splits = list(
            torch.split_with_sizes(pred_params, num_weight_params + num_bias_params, dim=1)
        )

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l == 0:
                # input layer
                weight_splits[l] = weight_splits[l].reshape(num_instances, channels, in_channels)
                bias_splits[l] = bias_splits[l].reshape(num_instances, channels, 1)
            elif l < num_layers - 1:
                # intermediate layer
                weight_splits[l] = weight_splits[l].reshape(num_instances, channels, channels)
                bias_splits[l] = bias_splits[l].reshape(num_instances, channels, 1)
            else:
                # output layer
                weight_splits[l] = weight_splits[l].reshape(num_instances, num_classes, channels)
                bias_splits[l] = bias_splits[l].reshape(num_instances, num_classes, 1)

        return weight_splits, bias_splits
    
class ConvFCHead(nn.Module):
    """
    A mask head with fully connected layers. Given pooled features it first reduces channels and
    spatial dimensions with conv layers and then uses FC layers to predict coarse masks analogously
    to the standard box head.
    """

    def __init__(
        self, input_shape, *, conv_dim, fc_dims, output_shape
    ):
        """
        Args:
            conv_dim: the output dimension of the conv layers
            fc_dims: a list of N>0 integers representing the output dimensions of N FC layers
            output_shape: shape of the output mask prediction
        """
        super().__init__()

        # fmt: off
        input_channels    = input_shape[0]
        input_h           = input_shape[1]
        input_w           = input_shape[2]
        self.output_shape = output_shape
        # fmt: on

        self.conv_layers = []
        if input_channels > conv_dim:
            self.reduce_channel_dim_conv = Conv2d(
                input_channels,
                conv_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
                activation=F.relu,
            )
            self.conv_layers.append(self.reduce_channel_dim_conv)

        self.reduce_spatial_dim_conv = Conv2d(
            conv_dim, conv_dim, kernel_size=2, stride=2, padding=0, bias=True, activation=F.relu
        )
        self.conv_layers.append(self.reduce_spatial_dim_conv)

        input_dim = conv_dim * input_h * input_w
        input_dim //= 4

        self.fcs = []
        for k, fc_dim in enumerate(fc_dims):
            fc = nn.Linear(input_dim, fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            input_dim = fc_dim

        output_dim = int(np.prod(self.output_shape))

        self.prediction = nn.Linear(fc_dims[-1], output_dim)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.prediction.weight, std=0.001)
        nn.init.constant_(self.prediction.bias, 0)

        for layer in self.conv_layers:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    @classmethod
    def from_config(cls, cfg, input_shape, output_shape=None):
        if output_shape is None:
            output_shape = (
                1,
                cfg.ROI_MASK_HEAD_OUTPUT_SIDE_RESOLUTION,
                cfg.ROI_MASK_HEAD_OUTPUT_SIDE_RESOLUTION,
            )
        fc_dim = cfg.ROI_MASK_HEAD_FC_DIM
        num_fc = cfg.ROI_MASK_HEAD_NUM_FC
        ret = dict(
            input_shape=input_shape,
            conv_dim=cfg.ROI_MASK_HEAD_CONV_DIM,
            fc_dims=[fc_dim] * num_fc,
            output_shape=output_shape,
        )
        return ret

    def forward(self, x):
        N = x.shape[0]
        for layer in self.conv_layers:
            x = layer(x)
        x = torch.flatten(x, start_dim=1)
        for layer in self.fcs:
            x = F.relu(layer(x))
        output_shape = [N] + list(self.output_shape)
        x = self.prediction(x)
        return x.view(*output_shape)#.flatten(1)
    
class PointRendMaskHead(nn.Module):
    def __init__(self, cfg, input_shape):
        super().__init__()
        self.scale = 1.0 / 4
        # point head
        self._init_point_head(cfg, input_shape)
        # coarse mask head
        self.roi_pooler_in_features = cfg.ROI_MASK_HEAD_IN_FEATURES
        self.roi_pooler_size = cfg.ROI_MASK_HEAD_POOLER_RESOLUTION
        in_channels = self.roi_pooler_in_features
        self._init_roi_head(
            cfg,
            [in_channels, self.roi_pooler_size, self.roi_pooler_size]
        )

    def _init_roi_head(self, cfg, input_shape):
        self.coarse_head = ConvFCHead(**ConvFCHead.from_config(cfg, input_shape))

    def _init_point_head(self, cfg, input_shape):
        # fmt: off
        self.mask_point_on                      = cfg.ROI_MASK_HEAD_POINT_HEAD_ON
        if not self.mask_point_on:
            return
        self.mask_point_in_features             = cfg.POINT_HEAD_IN_FEATURES
        self.mask_point_train_num_points        = cfg.POINT_HEAD_TRAIN_NUM_POINTS
        self.mask_point_oversample_ratio        = cfg.POINT_HEAD_OVERSAMPLE_RATIO
        self.mask_point_importance_sample_ratio = cfg.POINT_HEAD_IMPORTANCE_SAMPLE_RATIO
        # next three parameters are use in the adaptive subdivions inference procedure
        self.mask_point_subdivision_init_resolution = cfg.ROI_MASK_HEAD_OUTPUT_SIDE_RESOLUTION
        self.mask_point_subdivision_steps       = cfg.POINT_HEAD_SUBDIVISION_STEPS
        self.mask_point_subdivision_num_points  = cfg.POINT_HEAD_SUBDIVISION_NUM_POINTS
        # fmt: on

        in_channels = int(np.sum([input_shape[f].channels for f in self.mask_point_in_features]))
        self.point_head = ImplicitPointHead(cfg, [in_channels, 1, 1])

        # An optimization to skip unused subdivision steps: if after subdivision, all pixels on
        # the mask will be selected and recomputed anyway, we should just double our init_resolution
        while (
            4 * self.mask_point_subdivision_init_resolution**2
            <= self.mask_point_subdivision_num_points
        ):
            self.mask_point_subdivision_init_resolution *= 2
            self.mask_point_subdivision_steps -= 1

    def forward(self, features):
        """
        Args:
            features (dict[str, Tensor]): a dict of image-level features
            instances (list[Instances]): proposals in training; detected
                instances in inference
        """
        pass

    def _roi_pooler(self, features):
        """
        Extract per-box feature. This is similar to RoIAlign(sampling_ratio=1) except:
        1. It's implemented by point_sample
        2. It pools features across all levels and concat them, while typically
           RoIAlign select one level for every box. However in the config we only use
           one level (p2) so there is no difference.

        Returns:
            Tensor of shape (R, C, pooler_size, pooler_size) where R is the total number of boxes
        """

        B = len(features)
        output_size = self.roi_pooler_size
        point_coords = generate_regular_grid_point_coords(B, output_size, output_size, features.device)
        # For regular grids of points, this function is equivalent to `len(features_list)' calls
        # of `ROIAlign` (with `SAMPLING_RATIO=1`), and concat the results.
        roi_features, _ = point_sample_fine_grained_features(
            features, self.scale, point_coords
        )
        return roi_features.view(B, roi_features.shape[1], output_size, output_size)

    # def _sample_train_points(self, coarse_mask, instances):
    #     assert self.training
    #     gt_classes = torch.cat([x.gt_classes for x in instances])
    #     with torch.no_grad():
    #         # sample point_coords
    #         point_coords = get_uncertain_point_coords_with_randomness(
    #             coarse_mask,
    #             lambda logits: calculate_uncertainty(logits, gt_classes),
    #             self.mask_point_train_num_points,
    #             self.mask_point_oversample_ratio,
    #             self.mask_point_importance_sample_ratio,
    #         )
    #         # sample point_labels
    #         proposal_boxes = [x.proposal_boxes for x in instances]
    #         cat_boxes = Boxes.cat(proposal_boxes)
    #         point_coords_wrt_image = get_point_coords_wrt_image(cat_boxes.tensor, point_coords)
    #         point_labels = sample_point_labels(instances, point_coords_wrt_image)
    #     return point_coords, point_labels

    def _point_pooler(self, features, point_coords):
        # sample image-level features
        point_fine_grained_features, _ = point_sample_fine_grained_features(
            features, self.scale, point_coords
        )
        return point_fine_grained_features

    def _get_point_logits(self, point_fine_grained_features, point_coords, coarse_mask):
        coarse_features = point_sample(coarse_mask, point_coords, align_corners=False)
        point_logits = self.point_head(point_fine_grained_features, coarse_features)
        return point_logits

    def _subdivision_inference(self, features):
        assert not self.training

        mask_point_subdivision_num_points = features.shape[-2] * features.shape[-1] // 4
        mask_logits = None
        # +1 here to include an initial step to generate the coarsest mask
        # prediction with init_resolution, when mask_logits is None.
        # We compute initial mask by sampling on a regular grid. coarse_mask
        # can be used as initial mask as well, but it's typically very low-res
        # so it will be completely overwritten during subdivision anyway.
        for _ in range(self.mask_point_subdivision_steps + 1):
            if mask_logits is None:
                point_coords = generate_regular_grid_point_coords(
                    features.shape[0],
                    features.shape[-2]//2,
                    features.shape[-1]//2,
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
                    features.shape[-2]//2,
                    features.shape[-1]//2,
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


class ImplicitPointRendMaskHead(PointRendMaskHead):
    def __init__(self, cfg, input_shape, num_classes=1):
        self.flag = True
        self.num_classes = num_classes
        super().__init__(cfg, input_shape)

    def _init_roi_head(self, cfg, input_shape):
        pass
        # assert hasattr(self, "num_params"), "Please initialize point_head first!"
        # self.parameter_head = ConvFCHead(**ConvFCHead.from_config(cfg, input_shape, (self.num_params,)))
        # self.parameter_head.output_shape = (self.num_params,)

    def _init_point_head(self, cfg, input_shape):
        # fmt: off
        self.mask_point_on = True  # always on
        self.mask_point_train_num_points        = cfg.POINT_HEAD_TRAIN_NUM_POINTS
        # next two parameters are use in the adaptive subdivions inference procedure
        self.mask_point_subdivision_steps       = cfg.POINT_HEAD_SUBDIVISION_STEPS
        self.mask_point_subdivision_num_points  = cfg.POINT_HEAD_SUBDIVISION_NUM_POINTS
        # fmt: on

        in_channels = input_shape[0]
        self.point_head = ImplicitPointHead(cfg, [in_channels, 1, 1], self.num_classes)
        # self.num_params = self.point_head.num_params

        # inference parameters
        self.mask_point_subdivision_init_resolution = int(
            math.sqrt(self.mask_point_subdivision_num_points)
        )
        assert (
            self.mask_point_subdivision_init_resolution
            * self.mask_point_subdivision_init_resolution
            == self.mask_point_subdivision_num_points
        )

    def forward(self, features, skls=None, gt_masks=None):
        """
        Args:
            features: B C H W
        """
        if self.training:
            # parameters = self.parameter_head(self._roi_pooler(features))

            point_coords, point_labels = self._sample_train_points_with_skeleton(features, skls, gt_masks)
            point_fine_grained_features = self._point_pooler(features, point_coords)
            point_logits = self._get_point_logits(
                point_fine_grained_features, point_coords
            )

            return point_logits, point_labels
        else:
            # parameters = self.parameter_head(self._roi_pooler(features))
            return self._subdivision_inference(features)

    def _uniform_sample_train_points(self, features):
        assert self.training
        # uniform sample
        point_coords = torch.rand(
            len(features), self.mask_point_train_num_points, 2, device=features.device
        )
        # sample point_labels
        # point_coords_wrt_image = get_point_coords_wrt_image(point_coords)
        return point_coords
    
    def _sample_train_points_with_skeleton(self, features, skls, gt_masks):
        assert self.training

        hf = features.shape[-2]
        wf = features.shape[-1]

        h = int(hf // self.scale)
        w = int(wf // self.scale)

        y_mins = torch.min(skls[:, :, 1], dim=1)[0].type(torch.int)
        y_maxs = torch.max(skls[:, :, 1], dim=1)[0].type(torch.int)
        x_mins = torch.min(skls[:, :, 0], dim=1)[0].type(torch.int)
        x_maxs = torch.max(skls[:, :, 0], dim=1)[0].type(torch.int)

        y_mins = torch.clamp(y_mins - 10, min=0)
        y_maxs = torch.clamp(y_maxs + 10, max=h)
        x_mins = torch.clamp(x_mins - 10, min=0)
        x_maxs = torch.clamp(x_maxs + 10, max=w)

        point_coords = []
        point_labels = []
        for(i, (y_min, y_max, x_min, x_max)) in enumerate(zip(y_mins, y_maxs, x_mins, x_maxs)):
            num_samples = self.mask_point_train_num_points // 2
            neg_mask = torch.ones((h, w), dtype=torch.bool, device=features.device)
            neg_mask[y_min:y_max, x_min:x_max] = False
            if torch.sum(neg_mask) == 0:
                neg_mask[0, 0] = True
            neg_idxs = torch.nonzero(neg_mask).flip(1)
            neg_idxs = neg_idxs[torch.randperm(len(neg_idxs)),:][:num_samples]
            while len(neg_idxs) < num_samples:
                neg_idxs = torch.cat([neg_idxs, neg_idxs], dim=0)
            pos_mask = (gt_masks[i] > 0)
            pos_idxs = torch.nonzero(pos_mask).flip(1)
            while len(pos_idxs) < num_samples:
                pos_idxs = torch.cat([pos_idxs, pos_idxs], dim=0)
            pos_idxs = pos_idxs[torch.randperm(len(pos_idxs)),:][:num_samples]
            point_coords.append(torch.cat([neg_idxs, pos_idxs], dim=0))
            point_labels.append(torch.cat([torch.zeros(num_samples), torch.ones(num_samples)], dim=0))

            # if self.flag:
            #     plt.imsave(f'./neg_{i}.png', neg_mask.cpu().numpy())
            #     plt.imsave(f'./pos_{i}.png', pos_mask.cpu().numpy())
            #     self.flag = False


        point_coords = torch.stack(point_coords, dim=0).to(features.device)
        point_coords = point_coords / torch.tensor([w-1, h-1], dtype=torch.float, device=features.device).unsqueeze(0)
        point_labels = torch.stack(point_labels, dim=0).to(features.device)

        # print(point_coords.shape)
        return point_coords, point_labels


    def _get_point_logits(self, fine_grained_features, point_coords):
        return self.point_head(fine_grained_features, point_coords)
    
class JointMaskHead(ImplicitPointRendMaskHead):
    def __init__(self, cfg, input_shape, num_classes=1):
        super().__init__(cfg, input_shape, num_classes)

    def _sample_train_points_with_skeleton(self, features, skls, gt_masks):
        assert self.training

        hf = features.shape[-2]
        wf = features.shape[-1]

        h = int(hf // self.scale)
        w = int(wf // self.scale)

        y_mins = torch.min(skls[:, :, 1], dim=1)[0].type(torch.int)
        y_maxs = torch.max(skls[:, :, 1], dim=1)[0].type(torch.int)
        x_mins = torch.min(skls[:, :, 0], dim=1)[0].type(torch.int)
        x_maxs = torch.max(skls[:, :, 0], dim=1)[0].type(torch.int)

        y_mins = torch.clamp(y_mins - 10, min=0)
        y_maxs = torch.clamp(y_maxs + 10, max=h)
        x_mins = torch.clamp(x_mins - 10, min=0)
        x_maxs = torch.clamp(x_maxs + 10, max=w)

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

            # if self.flag:
            #     plt.imsave(f'./neg_{i}.png', neg_mask.cpu().numpy())
            #     # plt.imsave(f'./pos_{i}.png', pos_mask.cpu().numpy())
            #     self.flag = False

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

        # print(point_coords.shape)
        return point_coords, point_labels
    
if __name__ == '__main__':
    class cfg:
        POINT_HEAD_NUM_FC = 3
        POINT_HEAD_FC_DIM = 128
        POINT_HEAD_TRAIN_NUM_POINTS = 196
        POINT_HEAD_SUBDIVISION_STEPS = 3
        POINT_HEAD_SUBDIVISION_NUM_POINTS = 784
        IMPLICIT_POINTREND_IMAGE_FEATURE_ENABLED = True
        IMPLICIT_POINTREND_POS_ENC_ENABLED = True
        ROI_MASK_HEAD_OUTPUT_SIDE_RESOLUTION = 14
        ROI_MASK_HEAD_FC_DIM = 512
        ROI_MASK_HEAD_NUM_FC = 2
        ROI_MASK_HEAD_CONV_DIM = 128
        ROI_MASK_HEAD_IN_FEATURES = 256
        ROI_MASK_HEAD_POOLER_RESOLUTION = 14

    input_shape = [256, 14, 14]

    head = ImplicitPointRendMaskHead(cfg, input_shape)

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