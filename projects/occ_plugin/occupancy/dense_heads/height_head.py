# Developed by Jingyi Xu based on the codebase of Cam4DOcc and OpenOccupancy 
# Spatiotemporal Decoupling for Efficient Vision-Based Occupancy Forecasting
# https://github.com/BIT-XJY/EfficientOCF

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.core import reduce_mean
from mmdet.models import HEADS
from mmcv.cnn import build_conv_layer, build_norm_layer
from .lovasz_softmax import lovasz_softmax
from projects.occ_plugin.utils.nusc_param import nusc_class_names
from projects.occ_plugin.utils.semkitti import Smooth_L1_loss_for_height,L1_loss_for_height

@HEADS.register_module()
class HeightHead(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channel,
        num_level=1,
        num_img_level=1,
        soft_weights=False,
        conv_cfg=dict(type='Conv2d', bias=False),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        fine_topk=20000,
        empty_idx=0,
        balance_cls_weight=True,
        train_cfg=None,
        test_cfg=None,
    ):
        super(HeightHead, self).__init__()
        
        if type(in_channels) is not list:
            in_channels = [in_channels]
        
        self.in_channels = in_channels
        self.out_channel = out_channel
        self.num_level = num_level
        self.fine_topk = fine_topk
        
        # voxel-level prediction
        self.occ_convs = nn.ModuleList()
        for i in range(self.num_level):
            mid_channel = self.in_channels[i]
            occ_conv = nn.Sequential(
                build_conv_layer(conv_cfg, in_channels=self.in_channels[i], 
                        out_channels=mid_channel, kernel_size=3, stride=1, padding=1),
                build_norm_layer(norm_cfg, mid_channel)[1],
                nn.ReLU(inplace=True))
            self.occ_convs.append(occ_conv)

        self.occ_pred_conv = nn.Sequential(
                build_conv_layer(conv_cfg, in_channels=mid_channel, 
                        out_channels=mid_channel//2, kernel_size=1, stride=1, padding=0),
                build_norm_layer(norm_cfg, mid_channel//2)[1],
                nn.ReLU(inplace=True),)
        
        self.last_conv = build_conv_layer(conv_cfg, in_channels=mid_channel//2, 
                        out_channels=out_channel, kernel_size=1, stride=1, padding=0)
        self.last_conv.bias = nn.parameter.Parameter(torch.tensor([0.0], requires_grad=True))

        self.soft_weights = soft_weights
        self.num_img_level = num_img_level
        self.num_point_sampling_feat = self.num_level
        if self.soft_weights:
            soft_in_channel = mid_channel
            self.voxel_soft_weights = nn.Sequential(
                build_conv_layer(conv_cfg, in_channels=soft_in_channel, 
                        out_channels=soft_in_channel//2, kernel_size=1, stride=1, padding=0),
                build_norm_layer(norm_cfg, soft_in_channel//2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(conv_cfg, in_channels=soft_in_channel//2, 
                        out_channels=self.num_point_sampling_feat, kernel_size=1, stride=1, padding=0))
        
        if balance_cls_weight:
            self.class_weights = np.ones((out_channel,))
            self.class_weights[1:] = 5
            self.class_weights = torch.from_numpy(self.class_weights)
        else:
            self.class_weights = np.ones((out_channel,))

        self.class_names = nusc_class_names    
        self.empty_idx = empty_idx
        
    def forward_coarse_voxel(self, voxel_feats):
        output_occs = []
        output = {}
        for feats, occ_conv in zip(voxel_feats, self.occ_convs):
            output_occs.append(occ_conv(feats))
        
        if self.soft_weights:
            voxel_soft_weights = self.voxel_soft_weights(output_occs[0])
            voxel_soft_weights = torch.softmax(voxel_soft_weights, dim=1)
        else:
            voxel_soft_weights = torch.ones([output_occs[0].shape[0], self.num_point_sampling_feat, 1, 1, 1],).to(output_occs[0].device) / self.num_point_sampling_feat

        out_voxel_feats = 0
        _, _, H, W = output_occs[0].shape
        for feats, weights in zip(output_occs, torch.unbind(voxel_soft_weights, dim=1)): 
            feats = F.interpolate(feats, size=[H, W], mode='bilinear', align_corners=False).contiguous()
            out_voxel_feats += feats * weights.unsqueeze(1)
        out_voxel = self.occ_pred_conv(out_voxel_feats)
        out_voxel = self.last_conv(out_voxel)
        output['occ'] = [out_voxel]

        return output
    
    def forward(self, voxel_feats, img_feats=None, **kwargs):
        output = self.forward_coarse_voxel(voxel_feats)
        res = {'output_voxels': output['occ'],}
        
        return res

    def loss_height(self, output_height, target_height, tag):
        B, C, H, W = output_height.shape
        output_height = output_height[:,0,...].unsqueeze(-1).unsqueeze(-1)
        tB, tC, tH, tW = target_height.shape
        target_height = target_height.view(tB*tC, tH, tW) 
        ratio = target_height.shape[2] // H
        if ratio != 1:

            target_height = target_height.reshape(B, H, ratio, W, ratio).permute(0,1,3,2,4).reshape(B, H, W, ratio**2)
            empty_mask = target_height.sum(-1) == self.empty_idx
            target_height = target_height.to(torch.int64)
            occ_space = target_height[~empty_mask]
            occ_space[occ_space==0] = -torch.arange(len(occ_space[occ_space==0])).to(occ_space.device) - 1
            target_height[~empty_mask] = occ_space
            target_height = torch.mode(target_height, dim=-1)[0]
            target_height[target_height<0] = 0
        target_height = target_height.unsqueeze(-1).unsqueeze(-1)

        loss_dict = {}
        loss_dict['loss_height_l1_{}'.format(tag)] = (0.5) * Smooth_L1_loss_for_height(output_height, target_height, ignore_index=0)

        return loss_dict

    def loss(self, output_voxels=None,
                output_coords_fine=None, output_voxels_fine=None, 
                target_voxels=None, **kwargs):
        loss_dict = {}
        for index, output_voxel in enumerate(output_voxels):
            loss_dict.update(self.loss_height(output_voxel, target_voxels,  tag='c_{}'.format(index)))
            
        return loss_dict
