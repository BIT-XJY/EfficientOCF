import copy
from mmcv.runner import HOOKS, Hook
import time
try:
    from mmcv.cnn import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')
import torch
import torch.distributed as dist

@HOOKS.register_module()
class OccEfficiencyHook(Hook):
    def __init__(self, dataloader,  **kwargs):
        self.dataloader = dataloader 
        self.warm_up = 5
        
    def construct_input(self, DUMMY_SHAPE=None, m_info=None):
        if m_info is None:
            m_info = next(iter(self.dataloader))
        img_metas = m_info['img_metas'].data
        input = dict(
            img_metas=img_metas,
        )
        if 'img_inputs' in m_info.keys():
            img_inputs = m_info['img_inputs']
            for i in range(len(img_inputs)):
                if isinstance(img_inputs[i], list):
                    for j in range(len(img_inputs[i])):
                        img_inputs[i][j] = img_inputs[i][j].cuda()
                else:
                    img_inputs[i] = img_inputs[i].cuda()
            input['img_inputs'] = img_inputs
            
        if 'points' in m_info.keys():
            points = m_info['points'].data[0]
            points[0] = points[0].cuda()
            input['points'] = points
        return input
    
    def before_run(self, runner):
        torch.cuda.reset_peak_memory_stats()
        
        if dist.is_available() and dist.is_initialized():
            dist.barrier() 
        
    def after_run(self, runner):
        pass

    def before_epoch(self, runner):
        pass

    def after_epoch(self, runner):
        pass

    def before_iter(self, runner):
        pass

    def after_iter(self, runner):
        pass
