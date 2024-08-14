
# Traininig XoFTR

## Dataset setup
Generally, two parts of data are needed for training XoFTR, the original dataset, i.e., MegaDepth and KAIST Multispectral Pedestrian Detection Benchmark dataset. For MegaDepth the offline generated dataset indices are also required. The dataset indices store scenes, image pairs, and other metadata within the dataset used for training. For the MegaDepth dataset, the relative poses between images used for training are directly cached in the indexing files.

### Download datasets
#### MegaDepth
In the fine-tuning stage, we use depth maps, undistorted images, corresponding camera intrinsics and extrinsics provided in the [original MegaDepth dataset](https://www.cs.cornell.edu/projects/megadepth/).
- Please download [MegaDepth undistorted images and processed depths](https://www.cs.cornell.edu/projects/megadepth/dataset/Megadepth_v1/MegaDepth_v1.tar.gz)
    - The path of the download data will be referred to as `/path/to/megadepth`


#### KAIST Multispectral Pedestrian Detection Benchmark dataset
In the pre-training stage, we use LWIR and visible image pairs from [KAIST Multispectral Pedestrian Detection Benchmark](https://soonminhwang.github.io/rgbt-ped-detection/).

- Please set up the KAIST Multispectral Pedestrian Detection Benchmark dataset following [the official guide](https://github.com/SoonminHwang/rgbt-ped-detection) or from [OneDrive link](https://onedrive.live.com/download?cid=1570430EADF56512&resid=1570430EADF56512%21109419&authkey=AJcMP-7Yp86PWoE)
    - At the end, you should have the folder `kaist-cvpr15`, referred as `/path/to/kaist-cvpr15`

### Download the dataset indices

You can download the required dataset indices from the [following link](https://drive.google.com/drive/folders/1DOcOPZb3-5cWxLqn256AhwUVjBPifhuf).
After downloading, unzip the required files.
```shell
unzip downloaded-file.zip

# extract dataset indices
tar xf train-data/megadepth_indices.tar
```

### Build the dataset symlinks

We symlink the datasets to the `data` directory under the main XoFTR project directory.

```shell
# MegaDepth
# -- # fine-tuning dataset
ln -sv /path/to/megadepth/phoenix /path/to/XoFTR/data/megadepth/train
# -- # dataset indices
ln -s /path/to/megadepth_indices/* /path/to/XoFTR/data/megadepth/index

# KAIST Multispectral Pedestrian Detection Benchmark dataset
# -- # pre-training dataset
ln -sv /path/to/kaist-cvpr15 /path/to/XoFTR/data
```


## Training
We provide pre-training and fine-tuning scripts for the datasets. The results in the XoFTR paper can be reproduced with 2 RTX A5000 (24 GB) GPUs for pre-training and 8 A100 GPUs for fine-tuning. For a different setup, we scale the learning rate and its warm-up linearly, but the final evaluation results might vary due to the different batch size & learning rate used. Thus the reproduction of results in our paper is not guaranteed.


### Pre-training
``` shell
scripts/reproduce_train/pretrain.sh
```
> NOTE: Originally, we used 2 GPUs with a batch size of 2. You can change the number of GPUs and batch size in the script as per your need.

### Fine-tuning on MegaDepth
In the script, the path for pre-trained weights is `pretrain_weights/epoch=8-.ckpt`. We used the weight of the 9th epoch from the pre-training stage (epoch numbers start from 0). You can change this ckpt path accordingly.
``` shell
scripts/reproduce_train/visible_thermal.sh
```
> NOTE: Originally, we used 8 GPUs with a batch size of 2. You can change the number of GPUs and batch size in the script as per your need.