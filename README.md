# EfficientOCF

The official code for our paper: [**Spatiotemporal Decoupling for Efficient Vision-Based Occupancy Forecasting**](https://arxiv.org/abs/2411.14169).

This work has been accepted by CVPR 2025 :tada:

[Jingyi Xu#](https://github.com/BIT-XJY), [Xieyuanli Chen#](https://github.com/Chen-Xieyuanli), [Junyi Ma](https://github.com/BIT-MJY), Jiawei Huang, Jintao Xu, [Yue Wang](https://scholar.google.com.hk/citations?hl=zh-CN&user=N543LSoAAAAJ), [Ling Pei*](https://scholar.google.com.hk/citations?hl=zh-CN&user=Vm7d2EkAAAAJ).

![image](https://github.com/user-attachments/assets/9bdb9627-9edb-4d4f-a722-b7c99f49697e)


## Citation
If you use EfficientOCF in an academic work, please cite our paper:

	@inproceedings{xu2025cvpr,
		author = {Jingyi Xu and Xieyuanli Chen and Junyi Ma and Jiawei Huang and Jintao Xu and Yue Wang and Ling Pei},
		title = {{Spatiotemporal Decoupling for Efficient Vision-Based Occupancy Forecasting}},
		booktitle = {Proc.~of the IEEE/CVF Conf.~on Computer Vision and Pattern Recognition (CVPR)},
		year = 2025
	}


 ## Installation

<details>
	
<summary>We follow the installation instructions of our codebase [Cam4DOcc](https://github.com/haomo-ai/Cam4DOcc), which are also posted here
</summary>

* Create a conda virtual environment and activate it
```bash
conda create -n efficientocf python=3.7 -y
conda activate efficientocf
```
* Install PyTorch and torchvision (tested on torch==1.10.1 & cuda=11.3)
```bash
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```
* Install gcc>=5 in conda env
```bash
conda install -c omgarcia gcc-6
```
* Install mmcv, mmdet, and mmseg
```bash
pip install mmcv-full==1.4.0
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
```
* Install mmdet3d from the source code
```bash
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1 # Other versions may not be compatible.
python setup.py install
```
* Install other dependencies
```bash
pip install timm
pip install open3d-python
pip install PyMCubes
pip install spconv-cu113
pip install fvcore
pip install setuptools==59.5.0

pip install lyft_dataset_sdk # for lyft dataset
```
* Install occupancy pooling
```
git clone git@github.com:BIT-XJY/EfficientOCF.git
cd EfficientOCF
```

</details>

## Data Structure

### nuScenes dataset
* Please link your [nuScenes V1.0 full dataset](https://www.nuscenes.org/nuscenes#download) to the data folder. 
* [nuScenes-Occupancy](https://drive.google.com/file/d/1vTbgddMzUN6nLyWSsCZMb9KwihS7nPoH/view?usp=sharing), [nuscenes_occ_infos_train.pkl](https://github.com/JeffWang987/OpenOccupancy/releases/tag/train_pkl), and [nuscenes_occ_infos_val.pkl](https://github.com/JeffWang987/OpenOccupancy/releases/tag/val_pkl) are also provided by the previous work.
* [test_ids](https://drive.google.com/drive/folders/1O2nWOfpowNkad7_yq_Az2kKNnQ0HeeIu?usp=drive_link) are also needed for nuScenes dataset.

### Lyft dataset
* Please link your [Lyft dataset](https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles/data) to the data folder.
* The required folders are listed below:

```bash
EfficientOCF
├── data/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── lidarseg/
│   │   ├── v1.0-test/
│   │   ├── v1.0-trainval/
│   │   ├── nuscenes_occ_infos_train.pkl
│   │   ├── nuscenes_occ_infos_val.pkl
│   ├── nuScenes-Occupancy/
│   ├── lyft/
│   │   ├── maps/
│   │   ├── train_data/
│   │   ├── images/   # from train images, containing xxx.jpeg
│   ├── efficientocf
│   │   ├── GMO/
│   │   │   ├── ...
│   │   ├── GMO_lyft/
│   │   │   ├── ...
│   │   ├── test_ids/
```
Alternatively, you could manually modify the path parameters in the [config files](https://github.com/BIT-XJY/EfficientOCF/tree/main/projects/configs/baselines) instead of using the default data structure, which are also listed here:
```
occ_path = "./data/nuScenes-Occupancy"
depth_gt_path = './data/depth_gt'
train_ann_file = "./data/nuscenes/nuscenes_occ_infos_train.pkl"
val_ann_file = "./data/nuscenes/nuscenes_occ_infos_val.pkl"
ocf_dataset_path = "./data/efficientocf/"
nusc_root = './data/nuscenes/'
```

## Training and Evaluation

We directly integrate the EfficientOCF dataset generation pipeline into the dataloader, so you can directly run training or evaluate scripts and just wait


### Train with 8 GPUs

For the nuScenes or nuScenes-Occupancy datasets, please run

```bash
bash run.sh ./projects/configs/baselines/EfficientOCF_V1.1.py 8
```

For the Lyft dataset, please run

```bash
bash run.sh ./projects/configs/baselines/EfficientOCF_V1.1_lyft.py 8
```

### Test EfficientOCF for different tasks

If you only want to test the performance of occupancy prediction for the present frame (current observation), please set `test_present=True` in the [config files](https://github.com/BIT-XJY/EfficientOCF/tree/main/projects/configs/baselines). Otherwise, forecasting performance on the future interval is evaluated.

```bash
bash run_eval.sh $PATH_TO_CFG $PATH_TO_CKPT $GPU_NUM
# e.g. bash run_eval.sh ./projects/configs/baselines/EfficientOCF_V1.1.py ./work_dirs/EfficientOCF_V1.1/epoch_15.pth  8
```
Please set `save_pred` and `save_path` in the config files once saving prediction results are needed.

## Basic Information

Here is some basic information and key parameters for EfficientOCF.

| Type |  Info | Parameter |
| :----: | :----: | :----: |
| train           | 23,930 sequences | train_capacity |
| val             | 5,119 frames | test_capacity |
| voxel size      | 0.2m | voxel_x/y/z |
| range           | [-51.2m, -51.2m, -5m, 51.2m, 51.2m, 3m]| point_cloud_range |
| volume size     | [512, 512, 40]| occ_size |
| classes         | 2 for V1.1 / 9 for V1.2 | num_cls |
| observation frames | 3 | time_receptive_field |
| future frames | 4 | n_future_frames |
| extension frames | 6 | n_future_frames_plus |

Our proposed EfficientOCF can still perform well while being trained with partial data. Please try to decrease `train_capacity` if you want to explore more details with sparser supervision signals. 

In addition, please make sure that `n_future_frames_plus <= time_receptive_field + n_future_frames` because `n_future_frames_plus` means the real prediction number. We estimate more frames including the past ones rather than only `n_future_frames`.


### Acknowledgement

We thank the fantastic works [Cam4DOcc](https://github.com/haomo-ai/Cam4DOcc), [OpenOccupancy](https://github.com/JeffWang987/OpenOccupancy), [PowerBEV](https://github.com/EdwardLeeLPZ/PowerBEV), and [FIERY](https://anthonyhu.github.io/fiery) for their pioneer code release, which provide codebase for this benchmark.

