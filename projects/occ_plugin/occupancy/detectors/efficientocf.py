# Developed by Jingyi Xu based on the codebase of Cam4DOcc, OpenOccupancy and PowerBEV
# Spatiotemporal Decoupling for Efficient Vision-Based Occupancy Forecasting
# https://github.com/BIT-XJY/EfficientOCF

from sys import api_version
import torch
import collections 
import torch.nn.functional as F
import torch.nn as nn
import os
from mmdet.models import DETECTORS
from mmcv.runner import auto_fp16, force_fp32
from .bevdepth import BEVDepth
from mmdet3d.models import builder
import numpy as np
import time
import copy
from typing import Tuple

@DETECTORS.register_module()
class EfficientNet(BEVDepth):
    def __init__(self, 
            loss_cfg=None,
            only_generate_dataset=False,
            disable_loss_depth=False,
            test_present=False,
            empty_idx=0,
            max_label=2,
            occ_encoder_backbone=None,
            occ_predictor=None,
            occ_encoder_neck=None,
            height_encoder_backbone=None,
            height_predictor=None,
            height_encoder_neck=None,
            height_head=None,
            flow_encoder_backbone=None,
            flow_predictor=None,
            flow_encoder_neck=None,
            flow_head=None,
            loss_norm=False,
            point_cloud_range=None,
            time_receptive_field=None,
            n_future_frames=None,
            n_future_frames_plus=None,
            iou_thresh_for_vpq=None,
            record_time=False,
            save_pred=False,
            save_path=None,
            **kwargs):
        '''
        EfficientNet is our end-to-end baseline for 4D camera-only occupancy forecasting
        
        there are two streams for the forecasting task with aggregated voxel features as inputs:
            1. occ_encoder_backbone -> occ_predictor -> occ_encoder_neck -> pts_bbox_head
            2. flow_encoder_backbone -> flow_predictor -> flow_encoder_neck -> flow_head
        
        time_receptive_field: number of historical frames used for forecasting (including the present one), default: 3
        n_future_frames: number of forecasted future frames, default: 4
        n_future_frames_plus: number of estimated frames (> n_future_frames), default: 6 (if only forecasting occupancy states rather than instances, n_future_frames=n_future_frames_plus can be set)
        iou_thresh_for_vpq: iou threshold to associate instances in 3D instance prediction, default: 0.2 (adjusted by occupancy forecasting performance)
        '''
        super().__init__(**kwargs)

        self.loss_cfg = loss_cfg
        self.disable_loss_depth = disable_loss_depth
        self.only_generate_dataset = only_generate_dataset
        self.loss_norm = loss_norm
        self.time_receptive_field = time_receptive_field
        self.n_future_frames = n_future_frames
        self.n_future_frames_plus = n_future_frames_plus
        self.eval_start_moment = self.n_future_frames_plus - self.n_future_frames - 1

        self.iou_thresh_for_vpq = iou_thresh_for_vpq
        
        self.record_time = record_time
        self.time_stats = collections.defaultdict(list)
        self.empty_idx = empty_idx
        self.max_label = max_label

        self.occ_encoder_backbone = builder.build_backbone(occ_encoder_backbone)
        self.occ_predictor = builder.build_neck(occ_predictor)
        self.occ_encoder_neck = builder.build_neck(occ_encoder_neck)

        self.height_encoder_backbone = builder.build_backbone(height_encoder_backbone)
        self.height_predictor = builder.build_neck(height_predictor)
        self.height_encoder_neck = builder.build_neck(height_encoder_neck)
        self.height_head = builder.build_neck(height_head)

        self.flow_encoder_backbone = builder.build_backbone(flow_encoder_backbone)
        self.flow_encoder_neck = builder.build_neck(flow_encoder_neck)
        self.flow_predictor = builder.build_neck(flow_predictor)
        self.flow_head = builder.build_head(flow_head)

        self.point_cloud_range = point_cloud_range
        self.spatial_extent3d = (self.point_cloud_range[3]-self.point_cloud_range[0], \
                                    self.point_cloud_range[4]-self.point_cloud_range[1], \
                                         self.point_cloud_range[5]-self.point_cloud_range[2])
        self.ego_center_shift_proportion_x = abs(self.point_cloud_range[0])/(self.point_cloud_range[3]-self.point_cloud_range[0])
        self.ego_center_shift_proportion_y = abs(self.point_cloud_range[1])/(self.point_cloud_range[4]-self.point_cloud_range[1])
        self.ego_center_shift_proportion_z = abs(self.point_cloud_range[2])/(self.point_cloud_range[5]-self.point_cloud_range[2])

        self.n_cam = 6
        self.fine_grained = False
        self.vehicles_id = 1

        self.test_present = test_present
        self.save_pred = save_pred
        self.save_path = save_path

        self.mean_weight= nn.Parameter(torch.ones(1) * 0.1, requires_grad=True)
        self.max_weight= nn.Parameter(torch.ones(1) * 1.0, requires_grad=True)

    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        
        backbone_feats = self.img_backbone(imgs)

        if self.with_img_neck:
            x = self.img_neck(backbone_feats)
            if type(x) in [list, tuple]:
                x = x[0]
        else:
            x = backbone_feats
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        
        return {'x': x,
                'img_feats': [x.clone()]}
    
    @force_fp32()
    def occ_encoder(self, x):
        b, t, _, _, _ = x.shape
        x = x.reshape(b, -1, *x.shape[3:])
        x = self.occ_encoder_backbone(x)
        x = self.occ_predictor(x)
        x = self.occ_encoder_neck(x)

        return x
    
    @force_fp32()
    def height_encoder(self, x):
        b, t, _, _, _ = x.shape
        x = x.reshape(b, -1, *x.shape[3:])
        x = self.height_encoder_backbone(x)
        x = self.height_predictor(x)
        x = self.height_encoder_neck(x)

        return x

    @force_fp32()
    def flow_encoder(self, x):
        b, t, _, _, _ = x.shape
        x = x.reshape(b, -1, *x.shape[3:])
        x = self.flow_encoder_backbone(x)
        x = self.flow_predictor(x)
        x = self.flow_encoder_neck(x)
        return x

    def mat2pose_vec(self, matrix: torch.Tensor):
        """
        Converts a 4x4 pose matrix into a 6-dof pose vector
        Args:
            matrix (ndarray): 4x4 pose matrix
        Returns:
            vector (ndarray): 6-dof pose vector comprising translation components (tx, ty, tz) and
            rotation components (rx, ry, rz)
        """

        # M[1, 2] = -sinx*cosy, M[2, 2] = +cosx*cosy
        rotx = torch.atan2(-matrix[..., 1, 2], matrix[..., 2, 2])

        # M[0, 2] = +siny, M[1, 2] = -sinx*cosy, M[2, 2] = +cosx*cosy
        cosy = torch.sqrt(matrix[..., 1, 2] ** 2 + matrix[..., 2, 2] ** 2)
        roty = torch.atan2(matrix[..., 0, 2], cosy)

        # M[0, 0] = +cosy*cosz, M[0, 1] = -cosy*sinz
        rotz = torch.atan2(-matrix[..., 0, 1], matrix[..., 0, 0])

        rotation = torch.stack((rotx, roty, rotz), dim=-1)

        # Extract translation params
        translation = matrix[..., :3, 3]
        return torch.cat((translation, rotation), dim=-1)

    def pack_dbatch_and_dtime(self, x):
        b = x.shape[0]
        s = x.shape[1]
        x = x.view(b*s, *x.shape[2:])
        return x
    
    def unpack_dbatch_and_dtime(self, x, b, s):
        assert (b*s) == x.shape[0]
        x = x.view(b, s, *x.shape[1:])
        return x


    def extract_img_feat(self, img_inputs_seq, img_metas):
        '''
        Extract features of sequential input images
        '''
        
        imgs_seq, rots_seq, trans_seq, intrins_seq, post_rots_seq, post_trans_seq, gt_depths_seq, sensor2sensors_seq = img_inputs_seq

        self.batch_size = imgs_seq.shape[0]
        self.sequence_length = imgs_seq.shape[1]

        imgs_seq = imgs_seq[:,0:self.time_receptive_field,...].contiguous()
        rots_seq = rots_seq[:,0:self.time_receptive_field,...].contiguous()
        trans_seq = trans_seq[:,0:self.time_receptive_field,...].contiguous()
        intrins_seq = intrins_seq[:,0:self.time_receptive_field,...].contiguous()
        post_rots_seq = post_rots_seq[:,0:self.time_receptive_field,...].contiguous()
        post_trans_seq = post_trans_seq[:,0:self.time_receptive_field,...].contiguous()
        gt_depths_seq = gt_depths_seq[:,0:self.time_receptive_field,...].contiguous()
        sensor2sensors_seq = sensor2sensors_seq[:,0:self.time_receptive_field,...].contiguous()
        
        imgs_seq = self.pack_dbatch_and_dtime(imgs_seq)
        rots_seq = self.pack_dbatch_and_dtime(rots_seq)
        trans_seq = self.pack_dbatch_and_dtime(trans_seq)
        intrins_seq = self.pack_dbatch_and_dtime(intrins_seq)
        post_rots_seq = self.pack_dbatch_and_dtime(post_rots_seq)
        post_trans_seq = self.pack_dbatch_and_dtime(post_trans_seq)
        gt_depths_seq = self.pack_dbatch_and_dtime(gt_depths_seq)
        sensor2sensors_seq = self.pack_dbatch_and_dtime(sensor2sensors_seq)
        
        self.n_cam = imgs_seq.shape[1]
        
        img_enc_feats = self.image_encoder(imgs_seq)
        x = img_enc_feats['x']
        img_feats = img_enc_feats['img_feats']
        
        mlp_input_seq = self.img_view_transformer.get_mlp_input(rots_seq, trans_seq, intrins_seq, post_rots_seq, post_trans_seq)
        geo_inputs = [rots_seq, trans_seq, intrins_seq, post_rots_seq, post_trans_seq, None, mlp_input_seq]

        x, depth = self.img_view_transformer([x] + geo_inputs) 

        return x, depth, img_feats 
    
    def warp_features(self, x, flow, tseq):
        '''
        Warp features by motion flow
        '''
        if flow is None:
            return x

        b, dc, dx, dy, dz = x.shape

        # normalize 3D motion flow
        flow[:,0,-1] =flow[:,0,-1]*dx/self.spatial_extent3d[0]
        flow[:,1,-1] =flow[:,1,-1]*dy/self.spatial_extent3d[1]
        flow[:,2,-1] =flow[:,2,-1]*dz/self.spatial_extent3d[2]

        nx, ny, nz = torch.meshgrid(torch.arange(dx, dtype=torch.float, device=x.device), \
                                    torch.arange(dy, dtype=torch.float, device=x.device), \
                                    torch.arange(dz, dtype=torch.float, device=x.device))
        tmp = torch.ones((dx, dy, dz), device=x.device)
        grid = torch.stack((nx, ny, nz, tmp), dim=-1)

        # centralize shift
        shift_x = self.ego_center_shift_proportion_x * dx
        shift_y = self.ego_center_shift_proportion_y * dy
        shift_z = self.ego_center_shift_proportion_z * dz
        
        grid[:, :, :, 0] = grid[:, :, :, 0] - shift_x
        grid[:, :, :, 1] = grid[:, :, :, 1] - shift_y
        grid[:, :, :, 2] = grid[:, :, :, 2] - shift_z
        grid = grid.view(dx*dy*dz, grid.shape[-1]).unsqueeze(-1)

        transformation = flow.unsqueeze(1)
        transformed_grid = transformation @ grid
        transformed_grid = transformed_grid.squeeze(-1)
        transformed_grid = transformed_grid.view(-1, 4)

        # de-centralize
        transformed_grid[:, 0] = (transformed_grid[:, 0] + shift_x)
        transformed_grid[:, 1] = (transformed_grid[:, 1] + shift_y)
        transformed_grid[:, 2] = (transformed_grid[:, 2] + shift_z)
        transformed_grid = transformed_grid.round().long()

        # de-normalize
        grid = grid.squeeze(-1) 
        grid = grid.view(-1, 4)
        grid[:, 0] = (grid[:, 0] + shift_x)
        grid[:, 1] = (grid[:, 1] + shift_y)
        grid[:, 2] = (grid[:, 2] + shift_z)
        grid = grid.round().long()

        kept = (transformed_grid[:,0] >= 0) & (transformed_grid[:,0] <dx) \
               & (transformed_grid[:,1] >= 0) & (transformed_grid[:,1] <dy) \
               & (transformed_grid[:,2] >= 0) & (transformed_grid[:,2] < dz)

        transformed_grid = transformed_grid[kept]
        grid = grid[kept]

        warped_x =  torch.zeros_like(x, device=x.device)

        # loop for reduce memory cost
        interval_num = 32
        gap = transformed_grid.shape[0]//interval_num
        for tt in range(interval_num-1):
            start_idx_tt = int(tt*gap)
            end_idx_tt = int((tt+1)*gap)
            ixx = transformed_grid[start_idx_tt:end_idx_tt, 0]
            ixy = transformed_grid[start_idx_tt:end_idx_tt, 1]
            ixz = transformed_grid[start_idx_tt:end_idx_tt, 2]

            ixx_ori = grid[start_idx_tt:end_idx_tt, 0]
            ixy_ori = grid[start_idx_tt:end_idx_tt, 1]
            ixz_ori = grid[start_idx_tt:end_idx_tt, 2]

            warped_x[0, :, ixx, ixy, ixz] = x[0, :, ixx_ori, ixy_ori, ixz_ori]

        return warped_x 

    def cumulative_warp_occ(self, lifted_feature_seq, future_egomotion, mode='bilinear'):
        '''
        Warp sequential voxel features to the present frame by ego pose updaextract_feattes
        '''
        
        future_egomotion = future_egomotion[:, :self.time_receptive_field, ...].contiguous()
        
        out = [lifted_feature_seq[:, -1]]
        cum_future_egomotion = future_egomotion[:, -2]
        for t in reversed(range(self.time_receptive_field - 1)): 
            out.append(self.warp_features(lifted_feature_seq[:, t], cum_future_egomotion, t))
            cum_future_egomotion = cum_future_egomotion @ future_egomotion[:, t - 1]
        
        return torch.stack(out[::-1], 1)


    def extract_feat(self, img_inputs_seq, img_metas, future_egomotion):
        '''
        Extract voxel features from input sequential images
        '''
        voxel_feats = None
        depth, img_feats = None, None

        if img_inputs_seq is not None:
            voxel_feats, depth, img_feats = self.extract_img_feat(img_inputs_seq, img_metas)
        depth = depth.view(-1, self.n_cam, *depth.shape[-3:])
        
        voxel_feats = self.unpack_dbatch_and_dtime(voxel_feats, self.batch_size, self.time_receptive_field)

        voxel_feats = self.cumulative_warp_occ(voxel_feats.clone(), future_egomotion, mode='bilinear')

        # egomotion-aware
        future_egomotion_vec = self.mat2pose_vec(future_egomotion)
        batch_size, sequence_length, nbr_pose_channels = future_egomotion_vec.shape
        dx, dy, dz = voxel_feats.shape[-3:]
        
        future_egomotions_spatial = future_egomotion_vec.view(batch_size, sequence_length, nbr_pose_channels, 1, 1, 1).expand(batch_size, sequence_length, nbr_pose_channels, dx, dy, dz)
        
        # at time 0, no egomotion so feed zero vector
        future_egomotions_spatial = torch.cat([torch.zeros_like(future_egomotions_spatial[:, :1]),
                                            future_egomotions_spatial[:, :(self.time_receptive_field-1)]], dim=1)
        voxel_feats = torch.cat([voxel_feats, future_egomotions_spatial], dim=-4)
        
        max_feats = self.voxel2bev_maxpooling(voxel_feats) # max pooling
        mean_feats = self.voxel2bev(voxel_feats) # average pooling
        bev_feats = max_feats * self.max_weight + mean_feats * self.mean_weight

        bev_feats_enc = self.occ_encoder(bev_feats)
        if type(bev_feats_enc) is not tuple:
            bev_feats_enc = [bev_feats_enc]
        
        height_feats_enc = self.height_encoder(bev_feats)
        if type(height_feats_enc) is not tuple:
            height_feats_enc = [height_feats_enc]

        flow_feats_enc = self.flow_encoder(bev_feats)
        if type(flow_feats_enc) is not tuple:
            flow_feats_enc = [flow_feats_enc]
        
        return (bev_feats_enc, height_feats_enc, flow_feats_enc, img_feats, depth)
    
    def voxel2bev(self, voxel_feats):
        bev_feats = torch.mean(voxel_feats,-1)
        return bev_feats

    def voxel2bev_maxpooling(self, voxel_feats):
        bev_feats = torch.max(voxel_feats,-1).values
        return bev_feats
    
    @force_fp32(apply_to=('voxel_feats'))
    def forward_pts_train(
            self,
            voxel_feats,
            segmentation_bev=None,
            points_occ=None,
            img_metas=None,
            transform=None,
            img_feats=None,
        ):
        outs = self.pts_bbox_head(
            voxel_feats=voxel_feats,
            points=points_occ,
            img_metas=img_metas,
            img_feats=img_feats,
            transform=transform,)
        
        losses = self.pts_bbox_head.loss(
            output_voxels=outs['output_voxels'],
            target_voxels=segmentation_bev,
            target_points=points_occ,
            img_metas=img_metas,)
        
        return losses

    @force_fp32(apply_to=('voxel_feats'))
    def forward_height_train(
            self,
            voxel_feats,
            gt_height=None,
            points_occ=None,
            img_metas=None,
            transform=None,
            img_feats=None,
        ):

        outs = self.height_head(
            voxel_feats=voxel_feats,
            points=points_occ,
            img_metas=img_metas,
            img_feats=img_feats,
            transform=transform,
        )
        
        losses = self.height_head.loss(
            output_voxels=outs['output_voxels'],
            target_voxels=gt_height,
            target_points=points_occ,
            img_metas=img_metas,
        )
        
        return losses

    @force_fp32(apply_to=('voxel_feats'))
    def forward_flow_train(
            self,
            voxel_feats,
            gt_occ=None,
            points_occ=None,
            img_metas=None,
            transform=None,
            img_feats=None,
        ):

        outs = self.flow_head(
            voxel_feats=voxel_feats,
            points=points_occ,
            img_metas=img_metas,
            img_feats=img_feats,
            transform=transform,
        )
        
        losses = self.flow_head.loss(
            output_voxels=outs['output_voxels'],
            target_voxels=gt_occ,
            target_points=points_occ,
            img_metas=img_metas,
        )
        
        return losses

    def forward_train(self,
            img_inputs_seq=None,
            # segmentation=None,
            # instance=None,
            attribute_label=None,
            flow_bev=None,
            future_egomotion=None,
            gt_occ=None,
            height=None,
            segmentation_bev=None,
            instance_bev=None,
            img_metas=None,
            points_occ=None,
            **kwargs,
        ):
        '''
        Train EfficientNet using bbox-wise occupancy labels if self.fine_grained=False, else using voxel-wise labels from nuScenes-Occupancy
        '''
        # manually stop forward
        if self.only_generate_dataset:
            return {"pseudo_loss": torch.tensor(0.0, device=segmentation_bev.device, requires_grad=True)}

        voxel_feats, height_feats, flow_feats, img_feats, depth = self.extract_feat(
            img_inputs_seq=img_inputs_seq, img_metas=img_metas, future_egomotion=future_egomotion)
        
        height = height * 0.2
        
        height = height[:, -self.n_future_frames_plus:, ...]
        segmentation_bev = segmentation_bev[:, -self.n_future_frames_plus:, ...]
        flow_bev = flow_bev[:, -self.n_future_frames_plus:, ...]

        # training losses
        losses = dict()
        
        transform = img_inputs_seq[1:8] if img_inputs_seq is not None else None
        voxel_feats_seq = []
        for voxel_feats_stage in voxel_feats:
            bs, sfeatures = voxel_feats_stage.shape[:2]
            voxel_feats_stage_ = voxel_feats_stage.view(bs*self.n_future_frames_plus, sfeatures//self.n_future_frames_plus, *voxel_feats_stage.shape[2:])
            voxel_feats_seq.append(voxel_feats_stage_)

        # points_occ:None
        losses_occupancy = self.forward_pts_train(voxel_feats_seq, segmentation_bev,
                        points_occ, img_metas, img_feats=img_feats, transform=transform)
        losses.update(losses_occupancy)

        height_feats_seq = []
        for height_feats_stage in height_feats:
            bs, sfeatures = height_feats_stage.shape[:2]
            height_feats_stage_ = height_feats_stage.view(bs*self.n_future_frames_plus, sfeatures//self.n_future_frames_plus, *height_feats_stage.shape[2:])
            height_feats_seq.append(height_feats_stage_)
        
        # points_occ:None
        losses_height = self.forward_height_train(height_feats_seq, height, points_occ, img_metas, img_feats=img_feats, transform=transform)
        losses.update(losses_height)

        flow_feats_seq = []
        for flow_feats_stage in flow_feats:
            bs, sfeatures = flow_feats_stage.shape[:2]
            flow_feats_stage_ = flow_feats_stage.view(bs*self.n_future_frames_plus, sfeatures//self.n_future_frames_plus, *flow_feats_stage.shape[2:])
            flow_feats_seq.append(flow_feats_stage_)

        losses_flow = self.forward_flow_train(flow_feats_seq, flow_bev,
                        points_occ, img_metas, img_feats=img_feats, transform=transform)
        losses.update(losses_flow)

        if self.loss_norm:
            for loss_key in losses.keys():
                if loss_key.startswith('loss'):
                    losses[loss_key] = losses[loss_key] / (losses[loss_key].detach() + 1e-9)

        def logging_latencies():
            # logging latencies
            avg_time = {key: sum(val) / len(val) for key, val in self.time_stats.items()}
            sum_time = sum(list(avg_time.values()))
            out_res = ''
            for key, val in avg_time.items():
                out_res += '{}: {:.4f}, {:.1f}, '.format(key, val, val / sum_time)
            
            print(out_res)
        
        return losses
        
    def forward_test(self,
            img_inputs_seq=None,
            segmentation=None,
            # instance=None,
            attribute_label=None,
            flow_bev=None,
            future_egomotion=None,
            gt_occ=None,
            height=None,
            segmentation_bev=None,
            instance_bev=None,
            img_metas=None,
            points_occ=None,
            **kwargs,
        ):
        '''
        Test EfficientNet using IOU and VPQ metrics
        '''

        # let batch size equals 1 while testing
        # assert segmentation_bev.shape[0] == 1

        return self.simple_test(img_metas, img_inputs_seq, segmentation_bev=segmentation_bev, gt_height=height, gt_segmentation=segmentation, gt_occ=gt_occ, gt_flow=flow_bev, gt_instance=instance_bev, future_egomotion=future_egomotion, **kwargs)
    
    def simple_test(self, img_metas, img_inputs_seq=None, segmentation_bev=None, gt_height=None, gt_segmentation=None,
            gt_occ=None, gt_flow=None, gt_instance=None, future_egomotion=None, rescale=False, points_occ=None,):
        
        # manually stop forward
        if self.only_generate_dataset:
            return {'hist_for_iou': 0, 'pred_c': 0, 'vpq':0, 'height_l1':0, 'iou_3d':0, 'recall_3d':0}

        voxel_feats, height_feats, flow_feats, img_feats, depth = self.extract_feat(
            img_inputs_seq=img_inputs_seq, img_metas=img_metas, future_egomotion=future_egomotion)
        
        gt_height = gt_height * 0.2

        segmentation_bev = segmentation_bev[:, -self.n_future_frames_plus:, ...].contiguous()
        segmentation_bev = segmentation_bev.view(segmentation_bev.shape[0]*segmentation_bev.shape[1], *segmentation_bev.shape[2:])

        gt_segmentation = gt_segmentation[:, -self.n_future_frames_plus:, ...].contiguous()
        gt_segmentation = gt_segmentation.view(gt_segmentation.shape[0]*gt_segmentation.shape[1], *gt_segmentation.shape[2:])
        
        gt_height = gt_height[:, -self.n_future_frames_plus:, ...].contiguous()
        gt_height = gt_height.view(gt_height.shape[0]*gt_height.shape[1], *gt_height.shape[2:])

        gt_instance = gt_instance[:, -self.n_future_frames_plus:, ...].contiguous() 
        gt_instance = gt_instance.view(gt_instance.shape[0]*gt_instance.shape[1], *gt_instance.shape[2:])

        gt_flow = gt_flow[:, -self.n_future_frames_plus:, ...].contiguous()
        gt_flow = gt_flow.view(gt_flow.shape[0]*gt_flow.shape[1], *gt_flow.shape[2:])
        
        transform = img_inputs_seq[1:8] if img_inputs_seq is not None else None
        voxel_feats_seq = []
        for voxel_feats_stage in voxel_feats:
            bs, sfeatures = voxel_feats_stage.shape[:2]
            voxel_feats_stage_ = voxel_feats_stage.view(bs*self.n_future_frames_plus, sfeatures//self.n_future_frames_plus, *voxel_feats_stage.shape[2:])
            voxel_feats_seq.append(voxel_feats_stage_)
        
        output = self.pts_bbox_head(
            voxel_feats=voxel_feats_seq,
            points=points_occ,
            img_metas=img_metas,
            img_feats=img_feats,
            transform=transform,
        )

        pred_c = output['output_voxels'][0]  

        height_feats_seq = []
        for height_feats_stage in height_feats:
            bs, sfeatures = height_feats_stage.shape[:2]
            height_feats_stage_ = height_feats_stage.view(bs*self.n_future_frames_plus, sfeatures//self.n_future_frames_plus, *height_feats_stage.shape[2:])
            height_feats_seq.append(height_feats_stage_)
        
        output_height = self.height_head(
            voxel_feats=height_feats_seq,
            points=points_occ,
            img_metas=img_metas,
            img_feats=img_feats,
            transform=None,
        )
        pred_h = output_height['output_voxels'][0]
        
        flow_feats_seq = []
        for flow_feats_stage in flow_feats:
            bs, sfeatures = flow_feats_stage.shape[:2]
            flow_feats_stage_ = flow_feats_stage.view(bs*self.n_future_frames_plus, sfeatures//self.n_future_frames_plus, *flow_feats_stage.shape[2:])
            flow_feats_seq.append(flow_feats_stage_)

        output_flow = self.flow_head(
            voxel_feats=flow_feats_seq,
            points=points_occ,
            img_metas=img_metas,
            img_feats=img_feats,
            transform=transform,
        )

        pred_flow = output_flow['output_voxels'][0]

        vpq = self.evaluate_instance_prediction(pred_c, pred_flow, gt_instance, img_metas=img_metas, save_pred=self.save_pred, save_path=self.save_path)

        if self.test_present:
            pred_c = pred_c[self.eval_start_moment:(self.eval_start_moment+1), ...]
            segmentation_bev = segmentation_bev[self.eval_start_moment:(self.eval_start_moment+1), ...]
            gt_segmentation = gt_segmentation[self.eval_start_moment:(self.eval_start_moment+1), ...]
        else:
            pred_c = pred_c[self.eval_start_moment+1:, ...]
            segmentation_bev = segmentation_bev[self.eval_start_moment+1:, ...]
            gt_segmentation = gt_segmentation[self.eval_start_moment+1:, ...]

        hist_for_iou = self.evaluate_occupancy_forecasting(pred_c, segmentation_bev, img_metas=img_metas, save_pred=self.save_pred, save_path=self.save_path)

        if self.test_present:
            pred_h = pred_h[self.eval_start_moment:(self.eval_start_moment+1), ...]
            gt_height = gt_height[self.eval_start_moment:(self.eval_start_moment+1), ...]
        else:
            pred_h = pred_h[self.eval_start_moment+1:, ...]
            gt_height = gt_height[self.eval_start_moment+1:, ...]

        height_l1 = self.evaluate_height_forecasting(pred_h, gt_height, img_metas=img_metas, save_pred=self.save_pred, save_path=self.save_path)

        _, iou_3d, recall_3d = self.evaluate_3d_height(pred_c, segmentation_bev, pred_h, gt_height, gt_segmentation, img_metas=img_metas, save_pred=self.save_pred, save_path=self.save_path)

        test_output = {
            'hist_for_iou': hist_for_iou,
            'height_l1': height_l1,
            'pred_c': pred_c,
            'vpq': vpq,
            'iou_3d': iou_3d,
            'recall_3d': recall_3d,
        }
        return test_output
    
    def evaluate_height_forecasting(self, pred, gt, img_metas=None, save_pred=False, save_path=None):
        B, H, W = gt.shape
        pred = F.interpolate(pred, size=[H, W], mode='bilinear', align_corners=False).contiguous()
        pred = pred[:,0,...]
        pred[pred<0] = 0

        height_hist_list = []
        for k in range(B):
            pred_cur = pred[k].view(-1,1).cpu()
            gt_cur = gt[k].view(-1,1).cpu()
            kept = gt_cur[:,-1]!=0

            pred_cur = pred_cur[kept]
            gt_cur = gt_cur[kept]

            vpq = torch.abs(pred_cur-gt_cur)
            vpq_mean = vpq.mean()
            height_hist_list.append(vpq_mean)
        
        height_hist = sum(height_hist_list) / len(height_hist_list)

        save_height_path = os.path.join(save_path, 'height')
        if save_pred:
            if not os.path.exists(save_height_path):
                os.mkdir(save_height_path)
            pred_for_save_list = []
            for k in range(B):
                pred_height = pred[k]
                x_grid = torch.linspace(0, H-1, H, dtype=torch.long)
                x_grid = x_grid.view(H, 1).expand(H, W)
                y_grid = torch.linspace(0, W-1, W, dtype=torch.long)
                y_grid = y_grid.view(1, W).expand(H, W)

                height_for_save = torch.stack((x_grid, y_grid), -1)
                height_for_save = height_for_save.view(-1,2).cpu()
                height_label = pred_height.view(-1,1).cpu()
                height_for_save = torch.cat((height_for_save, height_label), dim=-1)
                kept = height_for_save[:,-1]!=0

                height_for_save= height_for_save[kept].cpu().numpy()
                pred_for_save_list.append(height_for_save)
            np.savez(os.path.join(save_height_path, img_metas[0]["scene_token"]), pred_for_save_list)
        
        return height_hist

    def evaluate_3d_height(self, pred_c, segmentation_bev, pred_h, gt_height, gt_segmentation, img_metas=None, save_pred=False, save_path=None):
        pred_h = pred_h / 0.2
        pred_h = pred_h.round()

        B, H, W = gt_height.shape
        pred_h = F.interpolate(pred_h, size=[H, W], mode='bilinear', align_corners=False).contiguous()
        pred_h = pred_h[:,0,...]
        pred_c = F.interpolate(pred_c, size=[H, W], mode='bilinear', align_corners=False).contiguous()
        pred_c = torch.argmax(pred_c, dim=1)

        grid_num = 40
        pred_3d = torch.zeros([B,H,W,grid_num])
        gt_3d = torch.zeros([B,H,W,grid_num])

        height_3d_diff = 0

        for k in range(B):
            pred_c_cur = pred_c[k].view(-1,1).cpu()
            pred_h_cur = pred_h[k].view(-1,1).cpu()
            kept = (pred_c_cur[:,-1]!=0) & (pred_h_cur[:,-1]!=0)

            height_new = torch.zeros_like(pred_h_cur)

            true_indices = torch.nonzero(kept).squeeze()
            true_indices_ist = true_indices.tolist()

            if true_indices_ist is None:
                continue
            if type(true_indices_ist) == int:
                true_indices_ist = [true_indices_ist]

            x_list = []
            y_list = []
            for true_idx in true_indices_ist:
                height_new[true_idx] = pred_h_cur[true_idx]
                x_idx = true_idx // H
                y_idx = true_idx % H
                x_list.append(x_idx)
                y_list.append(y_idx)
            height_new = height_new.view(H,W)

            for i in range(len(x_list)):
                x = x_list[i]
                y = y_list[i]
                h = int(height_new[x,y])
                pred_3d[k,x,y,:h] = 1

            segmentation_bev_cur = segmentation_bev[k].view(-1,1).cpu()
            gt_height_cur = gt_height[k].view(-1,1).cpu()
            kept = gt_height_cur[:,-1]!=0

            gt_height_new = torch.zeros_like(gt_height_cur)

            true_indices_gt = torch.nonzero(kept).squeeze()
            true_indices_gt_list = true_indices_gt.tolist()

            x_list = []
            y_list = []
            if true_indices_gt_list is None:
                continue
            if type(true_indices_gt_list) == int:
                true_indices_gt_list = [true_indices_gt_list]

            for true_idx in true_indices_gt_list:
                gt_height_new[true_idx] = gt_height_cur[true_idx]
                x_idx = true_idx // H
                y_idx = true_idx % H
                x_list.append(x_idx)
                y_list.append(y_idx)
            gt_height_new = gt_height_new.view(H,W)

            for i in range(len(x_list)):
                x = x_list[i]
                y = y_list[i]
                h = int(gt_height_new[x,y])
                gt_3d[k,x,y,:h] = 1

        height_3d_diff = torch.abs(pred_3d-gt_3d)
        height_3d_diff = height_3d_diff.mean()

        gt_3d_for_iou = copy.deepcopy(gt_3d.numpy())
        gt_3d_for_iou = gt_3d_for_iou.astype(int)
        gt_3d_for_iou = gt_3d_for_iou.flatten()
        pred_3d_for_iou = copy.deepcopy(pred_3d.numpy())
        pred_3d_for_iou = pred_3d_for_iou.astype(int)
        pred_3d_for_iou = pred_3d_for_iou.flatten()

        max_label = 2
        bin_count = np.bincount(max_label * gt_3d_for_iou.astype(int) + pred_3d_for_iou, minlength=max_label ** 2)

        kept_bbox = gt_segmentation!= 0
        gt_3d_bbox_for_iou = copy.deepcopy(gt_3d[kept_bbox].numpy())
        gt_3d_bbox_for_iou = gt_3d_bbox_for_iou.astype(int)
        gt_3d_bbox_for_iou = gt_3d_bbox_for_iou.flatten()
        pred_3d_bbox_for_iou = copy.deepcopy(pred_3d[kept_bbox].numpy())
        pred_3d_bbox_for_iou = pred_3d_bbox_for_iou.astype(int)
        pred_3d_bbox_for_iou = pred_3d_bbox_for_iou.flatten()

        bin_count_bbox = np.bincount(max_label * gt_3d_bbox_for_iou.astype(int) + pred_3d_bbox_for_iou, minlength=max_label ** 2)

        if bin_count is None:
            recall_3d = 0
            iou_3d = 0
        else:
            iou_3d = (bin_count[-1]/(bin_count[-1]+bin_count[1]+bin_count[2]))
            recall_3d = ((bin_count[-1]+bin_count_bbox[1])/(bin_count[-1]+bin_count[2]+bin_count[1]-bin_count_bbox[1])) 
        
        return height_3d_diff, iou_3d, recall_3d


    def evaluate_occupancy_forecasting(self, pred, gt, img_metas=None, save_pred=False, save_path=None):

        B, H, W = gt.shape
        pred = F.interpolate(pred, size=[H, W], mode='bilinear', align_corners=False).contiguous()

        hist_all = 0
        iou_per_pred_list = []
        pred_list = []
        gt_list = []
        for i in range(B):
            pred_cur = pred[i,...]
            pred_cur = torch.argmax(pred_cur, dim=0).cpu().numpy()
            gt_cur = gt[i, ...].cpu().numpy()
            gt_cur = gt_cur.astype(np.int)

            pred_list.append(pred_cur)
            gt_list.append(gt_cur)

            # ignore noise
            noise_mask = gt_cur != 255

            # GMO and others for max_label=2
            # multiple movable objects for max_label=9
            hist_cur, iou_per_pred = fast_hist(pred_cur[noise_mask], gt_cur[noise_mask], max_label=self.max_label)
            hist_all = hist_all + hist_cur
            iou_per_pred_list.append(iou_per_pred)

        # whether save prediction results
        save_seg_path = os.path.join(save_path, 'segmentation')
        if save_pred:
            if not os.path.exists(save_seg_path):
                os.mkdir(save_seg_path)
            pred_for_save_list = []
            for k in range(B):
                pred_for_save = torch.argmax(pred[k], dim=0).cpu()
                x_grid = torch.linspace(0, H-1, H, dtype=torch.long)
                x_grid = x_grid.view(H, 1).expand(H, W)
                y_grid = torch.linspace(0, W-1, W, dtype=torch.long)
                y_grid = y_grid.view(1, W).expand(H, W)
                segmentation_for_save = torch.stack((x_grid, y_grid), -1)
                segmentation_for_save = segmentation_for_save.view(-1, 2)
                segmentation_label = pred_for_save.squeeze(0).view(-1,1)
                segmentation_for_save = torch.cat((segmentation_for_save, segmentation_label), dim=-1)
                kept = segmentation_for_save[:,-1]!=0
                segmentation_for_save= segmentation_for_save[kept].cpu().numpy()
                pred_for_save_list.append(segmentation_for_save)
            np.savez(os.path.join(save_seg_path, img_metas[0]["scene_token"]), pred_for_save_list)

        return hist_all

    def find_instance_centers(self, center_prediction: torch.Tensor, conf_threshold: float = 0.1, nms_kernel_size: float = 3):
        
        assert len(center_prediction.shape) == 3

        center_prediction = F.threshold(center_prediction, threshold=conf_threshold, value=-1)

        nms_padding = (nms_kernel_size - 1) // 2
        maxpooled_center_prediction = F.max_pool2d(
            center_prediction, kernel_size=nms_kernel_size, stride=1, padding=nms_padding
        )

        # Filter all elements that are not the maximum (i.e. the center of the heatmap instance)
        center_prediction[center_prediction != maxpooled_center_prediction] = -1
        return torch.nonzero(center_prediction > 0)[:, 1:]

    def group_pixels(self, centers: torch.Tensor, offset_predictions: torch.Tensor) -> torch.Tensor:

        dx, dy = offset_predictions.shape[-2:]
        x_grid = (
            torch.arange(dx, dtype=offset_predictions.dtype, device=offset_predictions.device)
            .view(1, dx, 1)
            .repeat(1, 1, dy)
        )
        y_grid = (
            torch.arange(dy, dtype=offset_predictions.dtype, device=offset_predictions.device)
            .view(1, 1, dy)
            .repeat(1, dx, 1)
        )

        pixel_grid = torch.cat((x_grid, y_grid), dim=0)
        center_locations = (pixel_grid + offset_predictions).view(2, dx*dy, 1).permute(2, 1, 0)
        centers = centers.view(-1, 1, 2)

        distances = torch.norm(centers - center_locations, dim=-1)
        instance_id = torch.argmin(distances, dim=0).reshape(1, dx, dy) + 1
        return instance_id

    def update_instance_ids(self, instance_seg, old_ids, new_ids):
        indices = torch.arange(old_ids.max() + 1, device=instance_seg.device)
        for old_id, new_id in zip(old_ids, new_ids):
            indices[old_id] = new_id

        return indices[instance_seg].long()

    def make_instance_seg_consecutive(self, instance_seg):
        # Make the indices of instance_seg consecutive
        unique_ids = torch.unique(instance_seg)
        new_ids = torch.arange(len(unique_ids), device=instance_seg.device)
        instance_seg = self.update_instance_ids(instance_seg, unique_ids, new_ids)
        return instance_seg

    def get_instance_segmentation_and_centers(self,
        center_predictions: torch.Tensor,
        offset_predictions: torch.Tensor,
        foreground_mask: torch.Tensor,
        conf_threshold: float = 0.1,
        nms_kernel_size: float = 5,
        max_n_instance_centers: int = 100,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        dx, dy = offset_predictions.shape[-2:]
        center_predictions = center_predictions.view(1, dx, dy)
        offset_predictions = offset_predictions.view(2, dx, dy)
        foreground_mask = foreground_mask.view(1, dx, dy)

        centers = self.find_instance_centers(center_predictions, conf_threshold=conf_threshold, nms_kernel_size=nms_kernel_size)
        if not len(centers):
            return torch.zeros(center_predictions.shape, dtype=torch.int64, device=center_predictions.device)

        if len(centers) > max_n_instance_centers:
            centers = centers[:max_n_instance_centers].clone()
        
        instance_ids = self.group_pixels(centers, offset_predictions * foreground_mask.float())
        instance_seg = (instance_ids * foreground_mask.float()).long()

        # Make the indices of instance_seg consecutive
        instance_seg = self.make_instance_seg_consecutive(instance_seg)

        return instance_seg.long()

    def flow_warp(self, occupancy, flow, mode='nearest', padding_mode='zeros'):
        '''
        Warp ground-truth flow-origin occupancies according to predicted flows
        '''

        _, num_waypoints, _, grid_dx_cells, grid_dy_cells = occupancy.size()

        dx = torch.linspace(-1, 1, steps=grid_dx_cells)
        dy = torch.linspace(-1, 1, steps=grid_dy_cells)

        x_idx, y_idx = torch.meshgrid(dx, dy)
        identity_indices = torch.stack((x_idx, y_idx), dim=0).to(device=occupancy.device)

        warped_occupancy = []
        for k in range(num_waypoints):
            flow_origin_occupancy = occupancy[:, k]  # B T 1 dx dy -> B 1 dx dy 
            pred_flow = flow[:, k]  # B T 2 dx dy -> B 2 dx dy
            # Normalize along the width and height direction
            normalize_pred_flow = torch.stack(
                (2.0 * pred_flow[:, 0] / (grid_dx_cells - 1),  
                2.0 * pred_flow[:, 1] / (grid_dy_cells - 1)),
                dim=1,
            )

            warped_indices = identity_indices + normalize_pred_flow
            warped_indices = warped_indices.permute(0, 2, 3, 1)

            flow_origin_occupancy = flow_origin_occupancy.permute(0, 1, 3, 2)

            sampled_occupancy = F.grid_sample(
                input=flow_origin_occupancy,
                grid=warped_indices,
                mode=mode,
                padding_mode='zeros',
                align_corners=True,
            )
            warped_occupancy.append(sampled_occupancy)
        return warped_occupancy[0]

    def make_instance_id_temporally_consecutive(self, pred_inst, preds, backward_flow, ignore_index=255.0):

        assert pred_inst.shape[0] == 1, 'Assumes batch size = 1'

        # Initialise instance segmentations with prediction corresponding to the present
        consistent_instance_seg = [pred_inst.unsqueeze(0)]
        backward_flow = backward_flow.clone().detach()
        backward_flow[backward_flow == ignore_index] = 0.0
        seq_len, _, dx, dy = preds.shape

        for t in range(1, seq_len):

            init_warped_instance_seg = self.flow_warp(consistent_instance_seg[-1].unsqueeze(0).float(), backward_flow[t:t+1].unsqueeze(0)).int()

            warped_instance_seg = init_warped_instance_seg * preds[t:t+1, 0]

            consistent_instance_seg.append(warped_instance_seg)
        
        consistent_instance_seg = torch.cat(consistent_instance_seg, dim=1)
        return consistent_instance_seg

    def predict_instance_segmentation(self, pred_seg, pred_flow):

        pred_seg_sm = pred_seg.detach()
        pred_seg_sm = torch.argmax(pred_seg_sm, dim=1, keepdims=True)
        foreground_masks = pred_seg_sm.squeeze(1) == self.vehicles_id

        pred_inst_batch = self.get_instance_segmentation_and_centers(
            torch.softmax(pred_seg, dim=1)[0:1, self.vehicles_id].detach(),
            pred_flow[1:2].detach(), 
            foreground_masks[1:2].detach(),
            nms_kernel_size=7,
        )  
        
        consistent_instance_seg = self.make_instance_id_temporally_consecutive(
                pred_inst_batch,
                pred_seg_sm[1:],
                pred_flow[1:].detach(),
                )

        consistent_instance_seg = torch.cat([torch.zeros_like(pred_inst_batch.unsqueeze(0)), consistent_instance_seg], dim=1)

        return consistent_instance_seg.permute(1, 0, 2, 3).long()

    def combine_mask(self, segmentation: torch.Tensor, instance: torch.Tensor, n_classes: int, n_all_things: int):
        '''
        Shift all things ids by num_classes and combine things and stuff into a single mask
        '''
        instance = instance.view(-1)
        instance_mask = instance > 0
        instance = instance - 1 + n_classes

        segmentation = segmentation.clone().view(-1)
        segmentation_mask = segmentation < n_classes

        # Build an index from instance id to class id.
        instance_id_to_class_tuples = torch.cat(
            (
                instance[instance_mask & segmentation_mask].unsqueeze(1),
                segmentation[instance_mask & segmentation_mask].unsqueeze(1),
            ),
            dim=1,
        )

        instance_id_to_class = -instance_id_to_class_tuples.new_ones((n_all_things,))
        instance_id_to_class[instance_id_to_class_tuples[:, 0]] = instance_id_to_class_tuples[:, 1]
        instance_id_to_class[torch.arange(n_classes, device=segmentation.device)] = torch.arange(
            n_classes, device=segmentation.device
        )

        segmentation[instance_mask] = instance[instance_mask]
        segmentation += 1
        segmentation[~segmentation_mask] = 0

        return segmentation, instance_id_to_class

    def panoptic_metrics(self, pred_segmentation, pred_instance, gt_segmentation, gt_instance, unique_id_mapping):
        # GMO and others
        n_classes = 2 
        self.keys = ['iou', 'true_positive', 'false_positive', 'false_negative'] # hard coding
        result = {key: torch.zeros(n_classes, dtype=torch.float32, device=gt_instance.device) for key in self.keys}

        assert pred_segmentation.dim() == 2
        assert pred_segmentation.shape == pred_instance.shape == gt_segmentation.shape == gt_instance.shape

        n_instances = int(torch.cat([pred_instance, gt_instance]).max().item())
        n_all_things = n_instances + n_classes  # Classes + instances.
        n_things_and_void = n_all_things + 1

        pred_segmentation = pred_segmentation.long().detach().cpu()
        pred_instance = pred_instance.long().detach().cpu()
        gt_segmentation = gt_segmentation.long().detach().cpu()
        gt_instance = gt_instance.long().detach().cpu()
        
        prediction, pred_to_cls = self.combine_mask(pred_segmentation, pred_instance, n_classes, n_all_things)
        target, target_to_cls = self.combine_mask(gt_segmentation, gt_instance, n_classes, n_all_things)

        # Compute ious between all stuff and things
        # hack for bincounting 2 arrays together
        x = prediction + n_things_and_void * target  
        bincount_2d = torch.bincount(x.long(), minlength=n_things_and_void ** 2) 
        if bincount_2d.shape[0] != n_things_and_void ** 2:
            raise ValueError('Incorrect bincount size.')
        conf = bincount_2d.reshape((n_things_and_void, n_things_and_void))
        # Drop void class
        conf = conf[1:, 1:]  
        # Confusion matrix contains intersections between all combinations of classes
        union = conf.sum(0).unsqueeze(0) + conf.sum(1).unsqueeze(1) - conf
        iou = torch.where(union > 0, (conf.float() + 1e-9) / (union.float() + 1e-9), torch.zeros_like(union).float())

        mapping = (iou > self.iou_thresh_for_vpq).nonzero(as_tuple=False)
 
        # Check that classes match.
        is_matching = pred_to_cls[mapping[:, 1]] == target_to_cls[mapping[:, 0]]
        mapping = mapping[is_matching.detach().cpu().numpy()]
        tp_mask = torch.zeros_like(conf, dtype=torch.bool)
        tp_mask[mapping[:, 0], mapping[:, 1]] = True

        # First ids correspond to "stuff" i.e. semantic seg.
        # Instance ids are offset accordingly
        for target_id, pred_id in mapping:
            cls_id = pred_to_cls[pred_id]

            self.temporally_consistent = True # hard coding !
            if self.temporally_consistent and cls_id == self.vehicles_id:
                if target_id.item() in unique_id_mapping and unique_id_mapping[target_id.item()] != pred_id.item():
                    # Not temporally consistent
                    result['false_negative'][target_to_cls[target_id]] += 1
                    result['false_positive'][pred_to_cls[pred_id]] += 1
                    unique_id_mapping[target_id.item()] = pred_id.item()
                    continue

            result['true_positive'][cls_id] += 1
            result['iou'][cls_id] += iou[target_id][pred_id]
            unique_id_mapping[target_id.item()] = pred_id.item()

        for target_id in range(n_classes, n_all_things):
            # If this is a true positive do nothing.
            if tp_mask[target_id, n_classes:].any():
                continue
            # If this target instance didn't match with any predictions and was present set it as false negative.
            if target_to_cls[target_id] != -1:
                result['false_negative'][target_to_cls[target_id]] += 1

        for pred_id in range(n_classes, n_all_things):
            # If this is a true positive do nothing.
            if tp_mask[n_classes:, pred_id].any():
                continue
            # If this predicted instance didn't match with any prediction, set that predictions as false positive.
            if pred_to_cls[pred_id] != -1 and (conf[:, pred_id] > 0).any():
                result['false_positive'][pred_to_cls[pred_id]] += 1

        return result

    def evaluate_instance_prediction(self, pred_seg, pred_flow, gt_instance, img_metas, save_pred, save_path):

        B, H, W = gt_instance.shape
        h = 128
        w = 128

        pred_consistent_instance_seg = self.predict_instance_segmentation(pred_seg, pred_flow)

        # add one feature dimension for interpolate
        pred_consistent_instance_seg = F.interpolate(pred_consistent_instance_seg.float(), size=[H, W], mode='nearest').contiguous()
        pred_consistent_instance_seg = pred_consistent_instance_seg.squeeze(1)

        iou = 0
        true_positive = 0
        false_positive = 0
        false_negative = 0

        # starting from the present frame
        pred_instance = pred_consistent_instance_seg[self.eval_start_moment:]
        gt_instance = gt_instance[self.eval_start_moment:].long()

        assert gt_instance.min() == 0, 'ID 0 of gt_instance must be background'
        pred_segmentation = (pred_instance > 0).long()
        gt_segmentation = (gt_instance > 0).long()

        unique_id_mapping = {}
        for t in range(pred_segmentation.shape[0]):
            result = self.panoptic_metrics(
                pred_segmentation[t].detach(),
                pred_instance[t].detach(),
                gt_segmentation[t],
                gt_instance[t],
                unique_id_mapping,
            )

            iou += result['iou']
            true_positive += result['true_positive']
            false_positive += result['false_positive']
            false_negative += result['false_negative']

        denominator = torch.maximum(
            (true_positive + false_positive / 2 + false_negative / 2),
            torch.ones_like(true_positive)
        )
        pq = iou / denominator

        save_instance_path = os.path.join(save_path, 'instance')
        if save_pred:
            if not os.path.exists(save_instance_path):
                os.mkdir(save_instance_path)
            pred_for_save_list = []
            for k in range(B):
                pred_for_save = pred_consistent_instance_seg[k].cpu()
                x_grid = torch.linspace(0, H-1, H, dtype=torch.long)
                x_grid = x_grid.view(H, 1).expand(H, W)
                y_grid = torch.linspace(0, W-1, W, dtype=torch.long)
                y_grid = y_grid.view(1, W).expand(H, W)
                segmentation_for_save = torch.stack((x_grid, y_grid), -1)
                segmentation_for_save = segmentation_for_save.view(-1, 2)
                segmentation_label = pred_for_save.view(-1,1)
                segmentation_for_save = torch.cat((segmentation_for_save, segmentation_label), dim=-1)
                kept = segmentation_for_save[:,-1]!=0
                segmentation_for_save= segmentation_for_save[kept].cpu().numpy()
                pred_for_save_list.append(segmentation_for_save)
            np.savez(os.path.join(save_instance_path, img_metas[0]["scene_token"]), pred_for_save_list)
        
        return pq.cpu().numpy()

    def forward_dummy(self,
            points=None,
            img_metas=None,
            img_inputs=None,
            points_occ=None,
            **kwargs,
        ):

        voxel_feats, flow_feats, img_feats, depth = self.extract_feat(img=img_inputs, img_metas=img_metas)

        transform = img_inputs[1:8] if img_inputs is not None else None
        output = self.pts_bbox_head(
            voxel_feats=voxel_feats,
            points=points_occ,
            img_metas=img_metas,
            img_feats=img_feats,
            transform=transform,
        )
        
        return output
    
def fast_hist(pred, label, max_label=18):
    pred = copy.deepcopy(pred.flatten())
    label = copy.deepcopy(label.flatten())
    bin_count = np.bincount(max_label * label.astype(int) + pred, minlength=max_label ** 2) 
    iou_per_pred = (bin_count[-1]/(bin_count[-1]+bin_count[1]+bin_count[2]))
    return bin_count[:max_label ** 2].reshape(max_label, max_label),iou_per_pred