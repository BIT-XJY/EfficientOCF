# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.parallel import DataContainer as DC

from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import to_tensor
from mmdet3d.datasets.pipelines import DefaultFormatBundle3D

@PIPELINES.register_module()
class OccDefaultFormatBundle3D(DefaultFormatBundle3D):
    """Default formatting bundle.
    It simplifies the pipeline of formatting common fields for voxels,
    including "proposals", "gt_bboxes", "gt_labels", "gt_masks" and
    "gt_semantic_seg".
    These fields are formatted as follows.
    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        
    def __call__(self, results):
        """Call function to transform and format common fields in results.
        Args:
            results (dict): Result dict contains the data to convert.
        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        # Format 3D data
        results = super(OccDefaultFormatBundle3D, self).__call__(results)

        if 'segmentation' in results.keys():
            results['segmentation'] = DC(to_tensor(results['segmentation']), stack=True)
        if 'segmentation_bev' in results.keys():
            results['segmentation_bev'] = DC(to_tensor(results['segmentation_bev']), stack=True)
        if 'instance_bev' in results.keys():
            results['instance_bev'] = DC(to_tensor(results['instance_bev']), stack=True)
        if 'flow_bev' in results.keys():
            results['flow_bev'] = DC(to_tensor(results['flow_bev']), stack=True)
        if 'height' in results.keys():
            results['height'] = DC(to_tensor(results['height']), stack=True)

        return results