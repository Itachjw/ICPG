# **Image-Centered Pseudo Label Generation for Weakly Supervised Text-Based Person Re-Identification(Accepted By PRCV2024)**

## Authors: Weizhi Nie, Chengji Wang(*), Hao Sun, and Wei Xie

## Highlights

The goal of this work is to improve the performance of global text to image person retrieval under weakly supervised settings. To achieve this, we utilize the complete CLIP model as our feature extraction backbone. In addition, we propose an Image-Centered Pseudo Label Generation method to address the issues of uncertain cross-modal pseudo identity labels.

![](images/architecture.png)

## Usage

### Prepare Datasets
Download the CUHK-PEDES dataset from [here](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description), ICFG-PEDES dataset from [here](https://github.com/zifyloo/SSAN) and RSTPReid dataset form [here](https://github.com/NjtechCVLab/RSTPReid-Dataset)

Organize them in `your dataset root dir` folder as follows:
```
|-- your dataset root dir/
|   |-- <CUHK-PEDES>/
|       |-- imgs
|            |-- cam_a
|            |-- cam_b
|            |-- ...
|       |-- reid_raw.json
|
|   |-- <ICFG-PEDES>/
|       |-- imgs
|            |-- test
|            |-- train 
|       |-- ICFG_PEDES.json
|
|   |-- <RSTPReid>/
|       |-- imgs
|       |-- data_captions.json
```

## Training

```python
python train.py \
--name icpg \
--img_aug \
--batch_size 64 \
--MLM \
--loss_names 'chm+cdm+itc' \
--dataset_name 'CUHK-PEDES' \
--num_epoch 60
```

## Testing

```python
python test.py --config_file 'path/to/model_dir/configs.yaml'
```
