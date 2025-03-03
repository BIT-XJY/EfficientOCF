# Developed by Jingyi Xu based on the codebase of Cam4DOcc and PowerBEV 
# Spatiotemporal Decoupling for Efficient Vision-Based Occupancy Forecasting
# https://github.com/BIT-XJY/EfficientOCF

import numpy as np
from mmdet.datasets.builder import PIPELINES
import os
import torch
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
import time
import copy

@PIPELINES.register_module()
class LoadInstanceWithFlow(object):
    def __init__(self, ocf_dataset_path, grid_size=[512, 512, 40], pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], background=0, use_flow=True, use_separate_classes=False, use_lyft=False):
        '''
        Loading sequential occupancy labels and instance flows for training and testing
        '''
        self.ocf_dataset_path = ocf_dataset_path
        self.pc_range = pc_range
        self.resolution = [(self.pc_range[3+i] - self.pc_range[i])/grid_size[i] for i in range(len(self.pc_range[:3]))]
        self.start_position = [self.pc_range[i] + self.resolution[i] / 2.0 for i in range(len(self.pc_range[:3]))]
        self.dimension = grid_size
        self.pc_range = np.array(self.pc_range)
        self.resolution = np.array(self.resolution)
        self.start_position = np.array(self.start_position)
        self.dimension = np.array(self.dimension)
        self.background = background
        self.use_flow = use_flow
        self.use_separate_classes = use_separate_classes
        self.use_lyft = use_lyft

    def get_poly_region(self, instance_annotation, present_egopose, present_ego2lidar):
        """
        Obtain the bounding box polygon of the instance
        """
        present_ego_translation, present_ego_rotation = present_egopose
        present_ego2lidar_translation, present_ego2lidar_rotation = present_ego2lidar

        box = Box(
            instance_annotation['translation'], instance_annotation['size'], Quaternion(instance_annotation['rotation'])
        )
        box.translate(present_ego_translation)
        box.rotate(present_ego_rotation)

        box.translate(present_ego2lidar_translation)
        box.rotate(present_ego2lidar_rotation)
        pts=box.corners().T

        X_min_box = pts.min(axis=0)[0]
        X_max_box = pts.max(axis=0)[0]
        Y_min_box = pts.min(axis=0)[1]
        Y_max_box = pts.max(axis=0)[1]
        Z_min_box = pts.min(axis=0)[2]
        Z_max_box = pts.max(axis=0)[2]

        if self.pc_range[0] <= X_min_box and X_max_box <= self.pc_range[3] \
                and self.pc_range[1] <= Y_min_box and Y_max_box <= self.pc_range[4] \
                and self.pc_range[2] <= Z_min_box and Z_max_box <= self.pc_range[5]:
            pts = np.round((pts - self.start_position[:3] + self.resolution[:3] / 2.0) / self.resolution[:3]).astype(np.int32)

            return pts
        else:
            return None

    def fill_occupancy(self, occ_instance, occ_segmentation, occ_attribute_label, instance_fill_info):
        x_grid = torch.linspace(0, self.dimension[0]-1, self.dimension[0], dtype=torch.float)
        x_grid = x_grid.view(self.dimension[0], 1, 1).expand(self.dimension[0], self.dimension[1], self.dimension[2])
        y_grid = torch.linspace(0, self.dimension[1]-1, self.dimension[1], dtype=torch.float)
        y_grid = y_grid.view(1, self.dimension[1], 1).expand(self.dimension[0], self.dimension[1], self.dimension[2])
        z_grid = torch.linspace(0, self.dimension[2]-1, self.dimension[2], dtype=torch.float)
        z_grid = z_grid.view(1, 1, self.dimension[2]).expand(self.dimension[0], self.dimension[1], self.dimension[2])
        mesh_grid_3d = torch.stack((x_grid, y_grid, z_grid), -1)
        mesh_grid_3d = mesh_grid_3d.view(-1, 3)

        occ_instance = torch.from_numpy(occ_instance).view(-1, 1)
        occ_segmentation = torch.from_numpy(occ_segmentation).view(-1, 1)
        occ_attribute_label = torch.from_numpy(occ_attribute_label).view(-1, 1)
        occ_height = copy.deepcopy(occ_instance)

        for instance_info in instance_fill_info:
            poly_region_pts = instance_info['poly_region']
            semantic_id = instance_info['semantic_id']
            instance_id = instance_info['instance_id']
            attribute_label=instance_info['attribute_label']

            X_min_box = poly_region_pts.min(axis=0)[0]
            X_max_box = poly_region_pts.max(axis=0)[0]
            Y_min_box = poly_region_pts.min(axis=0)[1]
            Y_max_box = poly_region_pts.max(axis=0)[1]
            Z_min_box = poly_region_pts.min(axis=0)[2]
            Z_max_box = poly_region_pts.max(axis=0)[2]

            mask_cur_instance = (mesh_grid_3d[:,0] >= X_min_box) & (X_max_box >= mesh_grid_3d[:,0]) \
                                & (mesh_grid_3d[:,1] >= Y_min_box) & (Y_max_box >= mesh_grid_3d[:,1]) \
                                & (mesh_grid_3d[:,2] >= Z_min_box) & (Z_max_box >= mesh_grid_3d[:,2])
            occ_instance[mask_cur_instance] = instance_id
            occ_segmentation[mask_cur_instance] = semantic_id
            occ_attribute_label[mask_cur_instance] = attribute_label
            occ_height[mask_cur_instance] = Z_max_box
        
        occ_instance = occ_instance.view(self.dimension[0], self.dimension[1], self.dimension[2]).long()
        occ_segmentation = occ_segmentation.view(self.dimension[0], self.dimension[1], self.dimension[2]).long()
        occ_attribute_label = occ_attribute_label.view(self.dimension[0], self.dimension[1], self.dimension[2]).long()
        occ_height = occ_height.view(self.dimension[0], self.dimension[1], self.dimension[2]).long()

        if self.use_lyft:
            occ_height = torch.max(occ_height, -1).values

        return occ_instance, occ_segmentation, occ_attribute_label, occ_height

    def get_label(self, input_seq_data):
        """
        Generate labels for semantic segmentation, instance segmentation, z position, attribute from the raw data of nuScenes
        """
        timestep = self.counter
        # Background is ID 0
        segmentation = np.ones((self.dimension[0], self.dimension[1], self.dimension[2])) * self.background
        instance = np.ones((self.dimension[0], self.dimension[1], self.dimension[2])) * self.background
        attribute_label = np.ones((self.dimension[0], self.dimension[1], self.dimension[2]))  * self.background
        
        instance_dict = input_seq_data['instance_dict']
        egopose_list = input_seq_data['egopose_list']
        ego2lidar_list = input_seq_data['ego2lidar_list']
        time_receptive_field = input_seq_data['time_receptive_field']

        instance_fill_info = []
        
        for instance_token, instance_annotation in instance_dict.items():
            if timestep not in instance_annotation['timestep']:
                continue
            pointer = instance_annotation['timestep'].index(timestep)
            annotation = {
                'translation': instance_annotation['translation'][pointer],
                'rotation': instance_annotation['rotation'][pointer],
                'size': instance_annotation['size'],
            }
            
            poly_region = self.get_poly_region(annotation, egopose_list[time_receptive_field - 1], ego2lidar_list[time_receptive_field - 1]) 

            if isinstance(poly_region, np.ndarray):
                if self.counter >= time_receptive_field and instance_token not in self.visible_instance_set:
                    continue
                self.visible_instance_set.add(instance_token)

                prepare_for_fill = dict(
                    poly_region=poly_region,
                    instance_id=instance_annotation['instance_id'],
                    attribute_label=instance_annotation['attribute_label'][pointer],
                    semantic_id=instance_annotation['semantic_id'],
                )

                instance_fill_info.append(prepare_for_fill)

        instance, segmentation, attribute_label, bbox_height = self.fill_occupancy(instance, segmentation, attribute_label, instance_fill_info)

        segmentation = segmentation.unsqueeze(0)
        instance = instance.unsqueeze(0)
        attribute_label = attribute_label.unsqueeze(0).unsqueeze(0)
        bbox_height = bbox_height.unsqueeze(0)

        return segmentation, instance, attribute_label, bbox_height

    @staticmethod
    def generate_flow(flow, occ_instance_bev_seq, instance, instance_id):
        """
        Generate ground truth for the flow of each instance based on instance segmentation
        """
        seg_len, wx, wy = occ_instance_bev_seq.shape
        ratio = 4
        occ_instance_bev_seq = occ_instance_bev_seq.reshape(seg_len, wx//ratio, ratio, wy//ratio, ratio).permute(0,1,3,2,4).reshape(seg_len, wx//ratio, wy//ratio, ratio**2)
        empty_mask = occ_instance_bev_seq.sum(-1) == 0
        occ_instance_bev_seq = occ_instance_bev_seq.to(torch.int64)
        occ_space = occ_instance_bev_seq[~empty_mask]
        occ_space[occ_space==0] = -torch.arange(len(occ_space[occ_space==0])).to(occ_space.device) - 1 
        occ_instance_bev_seq[~empty_mask] = occ_space
        occ_instance_bev_seq = torch.mode(occ_instance_bev_seq, dim=-1)[0]
        occ_instance_bev_seq[occ_instance_bev_seq<0] = 0
        occ_instance_bev_seq = occ_instance_bev_seq.long()

        _, wx, wy = occ_instance_bev_seq.shape
        x, y = torch.meshgrid(torch.arange(wx, dtype=torch.float), torch.arange(wy, dtype=torch.float))
        
        grid = torch.stack((x, y), dim=0)

        # Set the first frame
        init_pointer = instance['timestep'][0]
        instance_mask = (occ_instance_bev_seq[init_pointer] == instance_id)

        flow[init_pointer, 0, instance_mask] = grid[0, instance_mask].mean(dim=0, keepdim=True).round() - grid[0, instance_mask]
        flow[init_pointer, 1, instance_mask] = grid[1, instance_mask].mean(dim=0, keepdim=True).round() - grid[1, instance_mask]

        for i, timestep in enumerate(instance['timestep']):
            if i == 0:
                continue

            instance_mask = (occ_instance_bev_seq[timestep] == instance_id)
            prev_instance_mask = (occ_instance_bev_seq[timestep-1] == instance_id)
            if instance_mask.sum() == 0 or prev_instance_mask.sum() == 0:
                continue

            flow[timestep, 0, instance_mask] = grid[0, prev_instance_mask].mean(dim=0, keepdim=True).round() - grid[0, instance_mask]
            flow[timestep, 1, instance_mask] = grid[1, prev_instance_mask].mean(dim=0, keepdim=True).round() - grid[1, instance_mask]

        return flow

    def get_flow_label(self, input_seq_data, ignore_index=255):
        """
        Generate the global map of the flow ground truth
        """
        occ_instance_bev = input_seq_data['instance_bev']
        instance_dict = input_seq_data['instance_dict']
        instance_map = input_seq_data['instance_map']

        seq_len, wx, wy = occ_instance_bev.shape
        ratio = 4
        flow = ignore_index * torch.ones(seq_len, 2, wx//ratio, wy//ratio)
        
        # ignore flow generation for faster pipelines
        if not self.use_flow:
            return flow

        for token, instance in instance_dict.items():
            flow = self.generate_flow(flow, occ_instance_bev, instance, instance_map[token])
        return flow.float()

    # set ignore index to 0 for vis
    @staticmethod
    def convert_instance_mask_to_center_and_offset_label(input_seq_data, ignore_index=255, sigma=3):
        occ_instance = input_seq_data['instance']
        num_instances=len(input_seq_data['instance_map'])

        seq_len, wx, wy, wz = occ_instance.shape
        center_label = torch.zeros(seq_len, 1, wx, wy, wz)
        offset_label = ignore_index * torch.ones(seq_len, 3, wx, wy, wz)
        # x is vertical displacement, y is horizontal displacement
        x, y, z = torch.meshgrid(torch.arange(wx, dtype=torch.float), torch.arange(wy, dtype=torch.float), torch.arange(wz, dtype=torch.float))
        
        # Ignore id 0 which is the background
        for instance_id in range(1, num_instances+1):
            for t in range(seq_len):
                instance_mask = (occ_instance[t] == instance_id)

                xc = x[instance_mask].mean().round().long()
                yc = y[instance_mask].mean().round().long()
                zc = z[instance_mask].mean().round().long()

                off_x = xc - x
                off_y = yc - y
                off_z = zc - z

                g = torch.exp(-(off_x ** 2 + off_y ** 2 + off_z ** 2) / sigma ** 2)
                center_label[t, 0] = torch.maximum(center_label[t, 0], g)
                offset_label[t, 0, instance_mask] = off_x[instance_mask]
                offset_label[t, 1, instance_mask] = off_y[instance_mask]
                offset_label[t, 2, instance_mask] = off_z[instance_mask]

        return center_label, offset_label

    def get_segmentation_bev(self, segmentation):
        segmentation = segmentation[0]
        x_voxel,y_voxel,z_voxel = segmentation.shape
        segmentation_bev = segmentation.sum(-1)
        segmentation_bev[segmentation_bev!=0] = 1
        return segmentation_bev

    def get_instance_bev(self, instance):
        instance = instance[0]
        x_voxel,y_voxel,z_voxel = instance.shape
        instance_bev = torch.zeros((x_voxel,y_voxel))

        for x in range(x_voxel):
            for y in range(y_voxel):
                sum_height = instance[x,y,:].view(-1)
                sum_height = torch.sum(sum_height)
                if sum_height != 0:
                    instance_bev[x,y] = self.non_zero_nbr(instance[x,y,:])
        return instance_bev

    def non_zero_nbr(self, instance_cur_xy):
        max_index = 0
        max_value = 0
        for i in range(instance_cur_xy.shape[0]):
            if instance_cur_xy[i] != 0:
                flag = 0
                for j in range(i+1,instance_cur_xy.shape[0]):
                    if instance_cur_xy[i] == instance_cur_xy[j]:
                        flag += 1
                if flag > max_index:
                    max_index = flag
                    max_value = i
        return instance_cur_xy[max_value]

    def __call__(self, results):
        assert 'attribute_label' not in results.keys()
        assert 'segmentation_bev' not in results.keys()
        assert 'instance_bev' not in results.keys()
        assert 'flow_bev' not in results.keys()

        time_receptive_field = results['time_receptive_field']

        prefix = "MMO" if self.use_separate_classes else "GMO"

        if self.use_lyft:
            prefix = prefix + "_lyft"

        seg_label_dir = os.path.join(self.ocf_dataset_path, prefix, "segmentation")
        if not os.path.exists(seg_label_dir):
            os.mkdir(seg_label_dir)
        seg_label_path = os.path.join(seg_label_dir, \
            results['input_dict'][time_receptive_field-1]['scene_token']+"_"+results['input_dict'][time_receptive_field-1]['lidar_token'])

        seg_bev_label_dir = os.path.join(self.ocf_dataset_path, prefix, "segmentation_bev")
        if not os.path.exists(seg_bev_label_dir):
            os.mkdir(seg_bev_label_dir)
        seg_bev_label_path = os.path.join(seg_bev_label_dir, \
            results['input_dict'][time_receptive_field-1]['scene_token']+"_"+results['input_dict'][time_receptive_field-1]['lidar_token'])

        instance_bev_label_dir = os.path.join(self.ocf_dataset_path, prefix, "instance_bev")
        if not os.path.exists(instance_bev_label_dir):
            os.mkdir(instance_bev_label_dir)
        instance_bev_label_path = os.path.join(instance_bev_label_dir, \
            results['input_dict'][time_receptive_field-1]['scene_token']+"_"+results['input_dict'][time_receptive_field-1]['lidar_token'])

        flow_bev_label_dir = os.path.join(self.ocf_dataset_path, prefix, "flow_bev")
        if not os.path.exists(flow_bev_label_dir):
            os.mkdir(flow_bev_label_dir)        
        flow_bev_label_path = os.path.join(flow_bev_label_dir, \
            results['input_dict'][time_receptive_field-1]['scene_token']+"_"+results['input_dict'][time_receptive_field-1]['lidar_token'])

        segmentation_list = []
        if os.path.exists(seg_label_path+".npz"):
            gt_segmentation_arr = np.load(seg_label_path+".npz",allow_pickle=True)['arr_0']
            for j in range(len(gt_segmentation_arr)):
                segmentation = np.zeros((self.dimension[0], self.dimension[1], self.dimension[2])) * self.background
                gt_segmentation = gt_segmentation_arr[j]
                gt_segmentation = torch.from_numpy(gt_segmentation)
                segmentation[gt_segmentation[:, 0].long(), gt_segmentation[:, 1].long(), gt_segmentation[:, 2].long()] = gt_segmentation[:, -1]
                segmentation = torch.from_numpy(segmentation).unsqueeze(0)
                segmentation_list.append(segmentation)

        segmentation_bev_list = []
        if os.path.exists(seg_bev_label_path+".npz"):
            gt_segmentation_bev_arr = np.load(seg_bev_label_path+".npz",allow_pickle=True)['arr_0']
            for j in range(len(gt_segmentation_bev_arr)):
                segmentation_bev = np.zeros((self.dimension[0], self.dimension[1])) * self.background
                gt_segmentation_bev = gt_segmentation_bev_arr[j]
                gt_segmentation_bev = torch.from_numpy(gt_segmentation_bev)
                segmentation_bev[gt_segmentation_bev[:, 0].long(), gt_segmentation_bev[:, 1].long()] = gt_segmentation_bev[:, -1]
                segmentation_bev = torch.from_numpy(segmentation_bev).unsqueeze(0)
                segmentation_bev_list.append(segmentation_bev)

        instance_bev_list = []
        if os.path.exists(instance_bev_label_path+".npz"):
            gt_instance_bev_arr = np.load(instance_bev_label_path+".npz",allow_pickle=True)['arr_0']
            for j in range(len(gt_instance_bev_arr)):
                instance_bev = np.ones((self.dimension[0], self.dimension[1])) * self.background
                gt_instance_bev = gt_instance_bev_arr[j]
                gt_instance_bev = torch.from_numpy(gt_instance_bev)
                instance_bev[gt_instance_bev[:, 0].long(), gt_instance_bev[:, 1].long()] = gt_instance_bev[:, -1]
                instance_bev = torch.from_numpy(instance_bev).unsqueeze(0)
                instance_bev_list.append(instance_bev)

        flow_bev_list = []
        if os.path.exists(flow_bev_label_path+".npz"):
            gt_flow_bev_arr = np.load(flow_bev_label_path+".npz",allow_pickle=True)['arr_0']
            for j in range(len(gt_flow_bev_arr)):
                flow_bev = np.ones((2, self.dimension[0]//4, self.dimension[1]//4)) * 255
                gt_flow_bev = gt_flow_bev_arr[j]
                gt_flow_bev = torch.from_numpy(gt_flow_bev)
                flow_bev[:, gt_flow_bev[:, 0].long(), gt_flow_bev[:, 1].long()] = gt_flow_bev[:, 2:].permute(1, 0)
                flow_bev = torch.from_numpy(flow_bev).unsqueeze(0)
                flow_bev_list.append(flow_bev)

        if self.use_lyft == True:
            pcd_height_label_dir = os.path.join(self.ocf_dataset_path, prefix, "pcd_height")
            if not os.path.exists(pcd_height_label_dir):
                os.mkdir(pcd_height_label_dir)        
            pcd_height_label_path = os.path.join(pcd_height_label_dir, \
                results['input_dict'][time_receptive_field-1]['scene_token']+"_"+results['input_dict'][time_receptive_field-1]['lidar_token'])

            pcd_height_list = []
            if os.path.exists(pcd_height_label_path+".npz"):
                gt_pcd_height_arr = np.load(pcd_height_label_path+".npz",allow_pickle=True)['arr_0']
                for j in range(len(gt_pcd_height_arr)):
                    pcd_height = np.zeros((self.dimension[0], self.dimension[1])) * self.background
                    gt_pcd_height = gt_pcd_height_arr[j]
                    gt_pcd_height = torch.from_numpy(gt_pcd_height)
                    pcd_height[gt_pcd_height[:, 0].long(), gt_pcd_height[:, 1].long()] = gt_pcd_height[:, -1]
                    pcd_height = torch.from_numpy(pcd_height).unsqueeze(0)
                    pcd_height_list.append(pcd_height)

        if os.path.exists(seg_label_path+".npz") and os.path.exists(seg_bev_label_path+".npz") and os.path.exists(instance_bev_label_path+".npz") and os.path.exists(flow_bev_label_path+".npz"):
            results['segmentation'] = torch.cat(segmentation_list, dim=0)
            # results['instance'] = torch.cat(instance_list, dim=0)
            results['attribute_label'] =  torch.from_numpy(np.zeros((self.dimension[0], self.dimension[1], self.dimension[2]))).unsqueeze(0)
            results['segmentation_bev'] = torch.cat(segmentation_bev_list, dim=0)
            results['instance_bev'] = torch.cat(instance_bev_list, dim=0)
            results['flow_bev'] = torch.cat(flow_bev_list, dim=0).float()

            if self.use_lyft == True:
                results['height'] = torch.cat(pcd_height_list, dim=0)
                for key, value in results.items():
                    if key in ['sample_token', 'centerness','flow_bev', 'instance', 'offset', 'time_receptive_field', "indices", \
                    'segmentation', 'segmentation_bev', 'instance_bev', 'height', 'attribute_label','sequence_length', 'instance_dict', 'instance_map', 'input_dict', 'egopose_list','ego2lidar_list','scene_token']:
                        continue
                    results[key] = torch.cat(value, dim=0)
                return results

            for key, value in results.items():
                if key in ['sample_token', 'centerness', 'offset', 'flow_bev', 'time_receptive_field', "indices", \
                    'segmentation', 'segmentation_bev', 'instance_bev', 'attribute_label','sequence_length', 'instance_dict', 'instance_map', 'input_dict', 'egopose_list','ego2lidar_list','scene_token']:
                    continue
                results[key] = torch.cat(value, dim=0)
            return results
        
        else:
            results['segmentation'] = []
            # results['instance'] = []
            results['attribute_label'] = []
            results['segmentation_bev'] = []
            results['instance_bev'] = []

            segmentation_saved_list = []
            # instance_saved_list = []
            segmentation_bev_saved_list = []
            instance_bev_saved_list = []

            if self.use_lyft:
                results['height'] = []
                pcd_height_saved_list = []

            sequence_length = results['sequence_length']
            self.visible_instance_set = set()
            for self.counter in range(sequence_length):
                segmentation, instance, attribute_label, bbox_height = self.get_label(results)
                segmentation_bev = self.get_segmentation_bev(segmentation)
                instance_bev = self.get_instance_bev(instance)
                
                results['segmentation'].append(segmentation)
                # results['instance'].append(instance)
                results['attribute_label'].append(attribute_label)
                results['segmentation_bev'].append(segmentation_bev.unsqueeze(0))
                results['instance_bev'].append(instance_bev.unsqueeze(0))

                x_grid = torch.linspace(0, self.dimension[0]-1, self.dimension[0], dtype=torch.long)
                x_grid = x_grid.view(self.dimension[0], 1, 1).expand(self.dimension[0], self.dimension[1], self.dimension[2])
                y_grid = torch.linspace(0, self.dimension[1]-1, self.dimension[1], dtype=torch.long)
                y_grid = y_grid.view(1, self.dimension[1], 1).expand(self.dimension[0], self.dimension[1], self.dimension[2])
                z_grid = torch.linspace(0, self.dimension[2]-1, self.dimension[2], dtype=torch.long)
                z_grid = z_grid.view(1, 1, self.dimension[2]).expand(self.dimension[0], self.dimension[1], self.dimension[2])
                segmentation_for_save = torch.stack((x_grid, y_grid, z_grid), -1)
                segmentation_for_save = segmentation_for_save.view(-1, 3)
                segmentation_label = segmentation.squeeze(0).view(-1,1)
                segmentation_for_save = torch.cat((segmentation_for_save, segmentation_label), dim=-1)
                kept = segmentation_for_save[:,-1]!=0
                segmentation_for_save= segmentation_for_save[kept]
                segmentation_saved_list.append(segmentation_for_save)

                x_grid_bev = torch.linspace(0, self.dimension[0]-1, self.dimension[0], dtype=torch.long)
                x_grid_bev = x_grid_bev.view(self.dimension[0], 1).expand(self.dimension[0], self.dimension[1])
                y_grid_bev = torch.linspace(0, self.dimension[1]-1, self.dimension[1], dtype=torch.long)
                y_grid_bev = y_grid_bev.view(1, self.dimension[1]).expand(self.dimension[0], self.dimension[1])
                
                segmentation_bev_for_save = torch.stack((x_grid_bev, y_grid_bev), -1)
                segmentation_bev_for_save = segmentation_bev_for_save.view(-1, 2)
                segmentation_bev_label = segmentation_bev.unsqueeze(-1).view(-1,1)
                segmentation_bev_for_save = torch.cat((segmentation_bev_for_save, segmentation_bev_label), dim=-1)
                kept = segmentation_bev_for_save[:,-1]!=0
                segmentation_bev_for_save= segmentation_bev_for_save[kept]
                segmentation_bev_saved_list.append(segmentation_bev_for_save)

                instance_bev_for_save = torch.stack((x_grid_bev, y_grid_bev), -1)
                instance_bev_for_save = instance_bev_for_save.view(-1, 2)
                instance_bev_label = instance_bev.unsqueeze(-1).view(-1,1)
                instance_bev_for_save = torch.cat((instance_bev_for_save, instance_bev_label), dim=-1)
                kept = instance_bev_for_save[:,-1]!=0
                instance_bev_for_save= instance_bev_for_save[kept]
                instance_bev_saved_list.append(instance_bev_for_save)

                if self.use_lyft:
                    results['height'].append(bbox_height)
                    x_grid_bev = torch.linspace(0, self.dimension[0]-1, self.dimension[0], dtype=torch.long)
                    x_grid_bev = x_grid_bev.view(self.dimension[0], 1).expand(self.dimension[0], self.dimension[1])
                    y_grid_bev = torch.linspace(0, self.dimension[1]-1, self.dimension[1], dtype=torch.long)
                    y_grid_bev = y_grid_bev.view(1, self.dimension[1]).expand(self.dimension[0], self.dimension[1])

                    pcd_height_for_save = torch.stack((x_grid_bev, y_grid_bev), -1)
                    pcd_height_for_save = pcd_height_for_save.view(-1, 2)
                    pcd_height_label = bbox_height.squeeze(0).view(-1,1)
                    pcd_height_for_save = torch.cat((pcd_height_for_save, pcd_height_label), dim=-1)
                    kept = pcd_height_for_save[:,-1]!=0
                    pcd_height_for_save= pcd_height_for_save[kept]
                    pcd_height_saved_list.append(pcd_height_for_save)
            
            segmentation_saved_list2 = [item.cpu().detach().numpy() for item in segmentation_saved_list]
            segmentation_bev_saved_list2 = [item.cpu().detach().numpy() for item in segmentation_bev_saved_list]
            # instance_saved_list2 = [item.cpu().detach().numpy() for item in instance_saved_list]
            instance_bev_saved_list2 = [item.cpu().detach().numpy() for item in instance_bev_saved_list]

            np.savez(seg_label_path, segmentation_saved_list2)
            np.savez(seg_bev_label_path, segmentation_bev_saved_list2)
            # np.savez(instance_label_path, instance_saved_list2)
            np.savez(instance_bev_label_path, instance_bev_saved_list2)

            if self.use_lyft:
                pcd_height_saved_list2 = [item.cpu().detach().numpy() for item in pcd_height_saved_list]
                np.savez(pcd_height_label_path, pcd_height_saved_list2)
                results['height'] = torch.cat(results['height'], dim=0)

            results['segmentation'] = torch.cat(results['segmentation'], dim=0)
            # results['instance'] = torch.cat(results['instance'], dim=0)
            results['attribute_label'] =  torch.from_numpy(np.zeros((self.dimension[0], self.dimension[1], self.dimension[2]))).unsqueeze(0)
            results['segmentation_bev'] = torch.cat(results['segmentation_bev'], dim=0)
            results['instance_bev'] = torch.cat(results['instance_bev'], dim=0)
            results['flow_bev'] = self.get_flow_label(results, ignore_index=255)
            flow_bev_saved_list = []
            sequence_length = results['sequence_length']
            d0 = self.dimension[0]//4
            d1 = self.dimension[1]//4 
            for cnt in range(sequence_length):
                flow_bev = results['flow_bev'][cnt, ...]
                x_grid = torch.linspace(0, d0-1, d0, dtype=torch.long)
                x_grid = x_grid.view(d0, 1).expand(d0, d1)
                y_grid = torch.linspace(0, d1-1, d1, dtype=torch.long)
                y_grid = y_grid.view(1, d1).expand(d0, d1)
                flow_bev_for_save = torch.stack((x_grid, y_grid), -1)
                flow_bev_for_save = flow_bev_for_save.view(-1, 2)
                flow_bev_label = flow_bev.permute(1,2,0).view(-1,2)
                flow_bev_for_save = torch.cat((flow_bev_for_save, flow_bev_label), dim=-1)
                kept = (flow_bev_for_save[:,-1]!=255) & (flow_bev_for_save[:,-2]!=255)
                flow_bev_for_save= flow_bev_for_save[kept]
                flow_bev_saved_list.append(flow_bev_for_save)

            flow_bev_saved_list2 = [item.cpu().detach().numpy() for item in flow_bev_saved_list]
            np.savez(flow_bev_label_path, flow_bev_saved_list2)

            if self.use_lyft:
                for key, value in results.items():
                    if key in ['sample_token', 'centerness', 'offset', 'flow_bev', 'time_receptive_field', "indices", 'height',\
                    'segmentation','segmentation_bev', 'instance_bev', 'attribute_label','sequence_length', 'instance_dict', 'instance_map', 'input_dict', 'egopose_list','ego2lidar_list','scene_token']:
                        continue
                    results[key] = torch.cat(value, dim=0)
            else:
                for key, value in results.items():
                    if key in ['sample_token', 'centerness', 'offset', 'flow_bev', 'time_receptive_field', "indices", \
                    'segmentation', 'segmentation_bev', 'instance_bev', 'attribute_label','sequence_length', 'instance_dict', 'instance_map', 'input_dict', 'egopose_list','ego2lidar_list','scene_token']:
                        continue
                    results[key] = torch.cat(value, dim=0)

        return results