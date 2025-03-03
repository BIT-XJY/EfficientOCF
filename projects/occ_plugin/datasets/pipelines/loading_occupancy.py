# Developed by Jingyi Xu based on the codebase of Cam4DOcc and PowerBEV 
# Spatiotemporal Decoupling for Efficient Vision-Based Occupancy Forecasting
# https://github.com/BIT-XJY/EfficientOCF

import numpy as np
import numba as nb
from mmdet.datasets.builder import PIPELINES
import yaml, os
import torch
import torch.nn.functional as F
import copy

@PIPELINES.register_module()
class LoadOccupancy(object):

    def __init__(self, to_float32=True, occ_path=None, ocf_dataset_path=None, grid_size=[512, 512, 40], pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], unoccupied=0, gt_resize_ratio=1, use_fine_occ=False, test_mode=False, use_lyft=False):
        '''
        Read sequential fine-grained occupancy labels from nuScenes-Occupancy if use_fine_occ=True
        '''
        self.to_float32 = to_float32
        self.occ_path = occ_path
        self.ocf_dataset_path = ocf_dataset_path
        self.grid_size = np.array(grid_size)
        self.unoccupied = unoccupied
        self.pc_range = np.array(pc_range)
        self.voxel_size = (self.pc_range[3:] - self.pc_range[:3]) / self.grid_size
        self.gt_resize_ratio = gt_resize_ratio
        self.use_fine_occ = use_fine_occ
        self.dimension = grid_size
        self.test_mode = test_mode
        self.use_lyft = use_lyft

    def get_seq_pseudo_occ(self, results):
        sequence_length = results['sequence_length']
        gt_occ_seq = []

        for count in range(sequence_length):
            processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.unoccupied
            processed_label = torch.from_numpy(processed_label)
            gt_occ_seq.append(processed_label)

        gt_occ_seq = torch.stack(gt_occ_seq)
        return gt_occ_seq

    def get_seq_occ(self, results, only_gt_occ=False):
        sequence_length = results['sequence_length']
        gt_occ_seq = []
        pcd_xyhl_list = []

        for count in range(sequence_length):

            if self.use_lyft == False:
                scene_token_cur = results['input_dict'][count]['scene_token']
                lidar_token_cur = results['input_dict'][count]['lidar_token']

                rel_path = 'scene_{0}/occupancy/{1}.npy'.format(scene_token_cur, lidar_token_cur)
                #  [z y x cls] or [z y x vx vy vz cls]
                pcd = np.load(os.path.join(self.occ_path, rel_path))
                pcd_label = pcd[..., -1:]
                pcd_label[pcd_label==0] = 255
                pcd_np_cor = self.voxel2world(pcd[..., [2,1,0]] + 0.5)
                untransformed_occ = copy.deepcopy(pcd_np_cor)

                egopose_list = results['egopose_list']
                ego2lidar_list = results['ego2lidar_list']
                time_receptive_field = results['time_receptive_field']
                present_global2ego = egopose_list[time_receptive_field - 1]
                present_ego2lidar = ego2lidar_list[time_receptive_field - 1]
                cur_global2ego = egopose_list[count]
                cur_ego2lidar = ego2lidar_list[count]

                pcd_np_cor = np.dot(cur_ego2lidar[1].inverse.rotation_matrix, pcd_np_cor.T)
                pcd_np_cor = pcd_np_cor.T
                pcd_np_cor = pcd_np_cor - cur_ego2lidar[0]
                # cur_ego -> global 
                pcd_np_cor = np.dot(cur_global2ego[1].inverse.rotation_matrix, pcd_np_cor.T)  
                pcd_np_cor = pcd_np_cor.T
                pcd_np_cor = pcd_np_cor - cur_global2ego[0]
                # global -> present_ego  
                pcd_np_cor = pcd_np_cor + present_global2ego[0]
                pcd_np_cor = np.dot(present_global2ego[1].rotation_matrix, pcd_np_cor.T)
                pcd_np_cor = pcd_np_cor.T
                # present_ego -> present_lidar
                pcd_np_cor = pcd_np_cor + present_ego2lidar[0]
                pcd_np_cor = np.dot(present_ego2lidar[1].rotation_matrix, pcd_np_cor.T) 
                pcd_np_cor = pcd_np_cor.T            

                pcd_np_cor = self.world2voxel(pcd_np_cor)

                # make sure the point is in the grid
                pcd_np_cor = np.clip(pcd_np_cor, np.array([0,0,0]), self.grid_size - 1)
                transformed_occ = copy.deepcopy(pcd_np_cor)
                pcd_np = np.concatenate([pcd_np_cor, pcd_label], axis=-1)

                if only_gt_occ == False:
                    pcd_np_filter = copy.deepcopy(pcd_np)
                    for otheridx in [0,1,7,8,11,12,13,14,15,16,17,18,255]:
                        pcd_np_filter[pcd_np_filter[:,-1]==otheridx,-1] = 0
                    for vehidx in [2,3,4,5,6,9,10]:
                        pcd_np_filter[pcd_np_filter[:,-1]==vehidx,-1] = 1
                
                    pcd_np_filter[:,0] = np.round(pcd_np_filter[:,0])
                    pcd_np_filter[:,1] = np.round(pcd_np_filter[:,1])
                    pcd_xyhl = np.zeros_like(pcd_np_filter)
                    height_bev = np.zeros((512,512))
                
                    xy_dict = {}
                    for i in range(len(pcd_np_filter)):
                        if pcd_np_filter[i][-1] != 0:
                            x = int(pcd_np_filter[i][0])
                            y = int(pcd_np_filter[i][1])
                            xy_pair = str(x) + '_' + str(y)
                            
                            if xy_pair in xy_dict.keys():
                                continue
                            else:
                                mask_cur = (pcd_np_filter[:,0] == x) & (pcd_np_filter[:,1] == y) & (pcd_np_filter[:,-1] != 0)
                                pcd_np_xy = pcd_np_filter[mask_cur]
                                pcd_np_xy = pcd_np_xy[:,2]
                                height_max = max(pcd_np_xy)
                                xy_dict[xy_pair] = height_max
                                
                                height_bev[x,y] = height_max
                            
            # 255: noise, 1-16 normal classes, 0 unoccupied
            # pcd_np = pcd_np[np.lexsort((pcd_np_cor[:, 0], pcd_np_cor[:, 1], pcd_np_cor[:, 2])), :]
            # pcd_np = pcd_np.astype(np.int64)
            # processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.unoccupied
            # processed_label = nb_process_label(processed_label, pcd_np)

            # processed_label = torch.from_numpy(processed_label)

            # # TODO: hard coding
            # for otheridx in [0,1,7,8,11,12,13,14,15,16,17,18,255]:
            #     processed_label[processed_label==otheridx] = 0
            # for vehidx in [2,3,4,5,6,9,10]:
            #     processed_label[processed_label==vehidx] = 1            
            
            processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.unoccupied
            gt_occ_seq.append(processed_label)
            if only_gt_occ == False:
                pcd_xyhl_list.append(torch.from_numpy(height_bev).unsqueeze(0))

        if only_gt_occ == False:
            return gt_occ_seq, pcd_xyhl_list
        else:
            return gt_occ_seq

    def __call__(self, results):
        
        if self.use_lyft == False:
            assert 'height' not in results.keys()

        time_receptive_field = results['time_receptive_field']

        prefix = "GMO"
        if self.use_lyft:
            prefix = prefix + "_lyft"
            results['gt_occ'] = self.get_seq_occ(results, only_gt_occ=True)
            return results

        height_bev_dir = os.path.join(self.ocf_dataset_path, prefix, "pcd_height")
        
        if not os.path.exists(height_bev_dir):
            os.mkdir(height_bev_dir)  
        height_bev_path = os.path.join(height_bev_dir, \
            results['input_dict'][time_receptive_field-1]['scene_token']+"_"+results['input_dict'][time_receptive_field-1]['lidar_token'])
        
        pcd_xyhl_list = []
        if os.path.exists(height_bev_path+".npz"):
            results['gt_occ'] = self.get_seq_occ(results, only_gt_occ=True)

            gt_pcd_xyhl_arr = np.load(height_bev_path+".npz",allow_pickle=True)['arr_0']

            for j in range(len(gt_pcd_xyhl_arr)):
                pcd_xyhl = np.zeros((self.dimension[0], self.dimension[1]))
                gt_pcd_xyhl = gt_pcd_xyhl_arr[j]
                gt_pcd_xyhl = torch.from_numpy(gt_pcd_xyhl)
                pcd_xyhl[gt_pcd_xyhl[:, 0].long(), gt_pcd_xyhl[:, 1].long()] = gt_pcd_xyhl[:, -1]
                pcd_xyhl = torch.from_numpy(pcd_xyhl).unsqueeze(0)
                pcd_xyhl_list.append(pcd_xyhl)

            results['height'] = torch.cat(pcd_xyhl_list, dim=0)
        
        else:
            results['gt_occ'], pcd_xyhl_saved_list = self.get_seq_occ(results, only_gt_occ=False)
            results['height'] = torch.cat(pcd_xyhl_saved_list, dim=0)

            sequence_length = results['sequence_length']
            height_bev_saved_list = []
            for self.counter in range(sequence_length):
                height_bev = pcd_xyhl_saved_list[self.counter]

                x_grid_bev = torch.linspace(0, self.dimension[0]-1, self.dimension[0], dtype=torch.long)
                x_grid_bev = x_grid_bev.view(self.dimension[0], 1).expand(self.dimension[0], self.dimension[1])
                y_grid_bev = torch.linspace(0, self.dimension[1]-1, self.dimension[1], dtype=torch.long)
                y_grid_bev = y_grid_bev.view(1, self.dimension[1]).expand(self.dimension[0], self.dimension[1])
                
                height_bev_for_save = torch.stack((x_grid_bev, y_grid_bev), -1)
                height_bev_for_save = height_bev_for_save.view(-1, 2)
                height_bev_label = height_bev.view(-1,1)
                height_bev_for_save = torch.cat((height_bev_for_save, height_bev_label), dim=-1)
                kept = height_bev_for_save[:,-1]!=0
                height_bev_for_save= height_bev_for_save[kept]
                height_bev_saved_list.append(height_bev_for_save)

            height_bev_saved_list2 = [item.cpu().detach().numpy() for item in height_bev_saved_list]
            np.savez(height_bev_path, height_bev_saved_list2)

        # print('Loading occupancy finish!')

        return results

    def voxel2world(self, voxel):
        """
        voxel: [N, 3]
        """
        return voxel * self.voxel_size[None, :] + self.pc_range[:3][None, :]

    def world2voxel(self, world):
        """
        world: [N, 3]
        """
        return (world - self.pc_range[:3][None, :]) / self.voxel_size[None, :]

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}'
        return repr_str

    def project_points(self, points, rots, trans, intrins, post_rots, post_trans):
        
        # from lidar to camera
        points = points.reshape(-1, 1, 3)
        points = points - trans.reshape(1, -1, 3)
        inv_rots = rots.inverse().unsqueeze(0)
        points = (inv_rots @ points.unsqueeze(-1))
        
        # from camera to raw pixel
        points = (intrins.unsqueeze(0) @ points).squeeze(-1)
        points_d = points[..., 2:3]
        points_uv = points[..., :2] / points_d
        
        # from raw pixel to transformed pixel
        points_uv = post_rots[:, :2, :2].unsqueeze(0) @ points_uv.unsqueeze(-1)
        points_uv = points_uv.squeeze(-1) + post_trans[..., :2].unsqueeze(0)
        points_uvd = torch.cat((points_uv, points_d), dim=2)
        
        return points_uvd
    
# b1:boolean, u1: uint8, i2: int16, u2: uint16
@nb.jit('b1[:](i2[:,:],u2[:,:],b1[:])', nopython=True, cache=True, parallel=False)
def nb_process_img_points(basic_valid_occ, depth_canva, nb_valid_mask):
    # basic_valid_occ M 3
    # depth_canva H W
    # label_size = M   # for original occ, small: 2w mid: ~8w base: ~30w
    canva_idx = -1 * np.ones_like(depth_canva, dtype=np.int16)
    for i in range(basic_valid_occ.shape[0]):
        occ = basic_valid_occ[i]
        if occ[2] < depth_canva[occ[1], occ[0]]:
            if canva_idx[occ[1], occ[0]] != -1:
                nb_valid_mask[canva_idx[occ[1], occ[0]]] = False

            canva_idx[occ[1], occ[0]] = i
            depth_canva[occ[1], occ[0]] = occ[2]
            nb_valid_mask[i] = True
    return nb_valid_mask

# u1: uint8, u8: uint16, i8: int64
@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label_withvel(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    
    return processed_label

# u1: uint8, u8: uint16, i8: int64
@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    
    return processed_label