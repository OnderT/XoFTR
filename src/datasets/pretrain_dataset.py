import os
import glob
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from loguru import logger
import random
from src.utils.dataset import read_pretrain_gray

class PretrainDataset(Dataset):
    def __init__(self,
                 root_dir,
                 mode='train',
                 img_resize=None,
                 df=None,
                 img_padding=False,
                 frame_gap=2,
                 **kwargs):
        """
        Manage image pairs of KAIST Multispectral Pedestrian Detection Benchmark Dataset.
        
        Args:
            root_dir (str): KAIST Multispectral Pedestrian  root directory that has `phoenix`.
            mode (str): options are ['train', 'val']
            img_resize (int, optional): the longer edge of resized images. None for no resize. 640 is recommended.
                                        This is useful during training with batches and testing with memory intensive algorithms.
            df (int, optional): image size division factor. NOTE: this will change the final image size after img_resize.
            img_padding (bool): If set to 'True', zero-pad the image to squared size. This is useful during training.
            augment_fn (callable, optional): augments images with pre-defined visual effects.
        """
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode

        # specify which part of the data is used for trainng and testing
        if mode == 'train':
            assert img_resize is not None and img_padding 
            self.start_ratio = 0.0
            self.end_ratio = 0.9
        elif mode == 'val':
            assert img_resize is not None and img_padding 
            self.start_ratio = 0.9
            self.end_ratio = 1.0
        else:
            raise NotImplementedError()
        
        # parameters for image resizing, padding 
        self.img_resize = img_resize
        self.df = df
        self.img_padding = img_padding

        # for training XoFTR
        self.coarse_scale = getattr(kwargs, 'coarse_scale', 0.125)

        self.pair_paths = self.generate_kaist_pairs(root_dir, frame_gap=frame_gap, second_frame_range=0)

    def get_kaist_image_paths(self, root_dir):
        vis_img_paths = []
        lwir_img_paths = []
        img_num_per_folder = []

        # Recursively search for folders named "image"
        for folder, subfolders, filenames in os.walk(root_dir):
            if "visible" in subfolders and "lwir" in subfolders:
                vis_img_folder = osp.join(folder, "visible")
                lwir_img_folder = osp.join(folder, "lwir")
                # Use glob to find image files (you can add more extensions if needed)
                vis_imgs_i = glob.glob(osp.join(vis_img_folder, '*.jpg'))
                vis_imgs_i.sort()
                lwir_imgs_i = glob.glob(osp.join(lwir_img_folder, '*.jpg'))      
                lwir_imgs_i.sort()     
                vis_img_paths.append(vis_imgs_i)
                lwir_img_paths.append(lwir_imgs_i)
                img_num_per_folder.append(len(vis_imgs_i))
                assert len(vis_imgs_i) == len(lwir_imgs_i), f"Image numbers do not match in {folder}, {len(vis_imgs_i)} != {len(lwir_imgs_i)}"
                # Add more image file extensions as necessary
        return vis_img_paths, lwir_img_paths, img_num_per_folder
    
    def generate_kaist_pairs(self, root_dir, frame_gap, second_frame_range):
        """ generate image pairs (Vis-TIR) from KAIST Pedestrian dataset
        Args:
            root_dir: root directory for the dataset
            frame_gap (int): the frame gap between consecutive images 
            second_frame_range (int): the range for second image i.e. for the first ind i, second ind j element of [i-10, i+10]
        Returns:
            pair_paths (list)
        """
        vis_img_paths, lwir_img_paths, img_num_per_folder = self.get_kaist_image_paths(root_dir)
        pair_paths = []
        for i in range(len(img_num_per_folder)):
            num_img = img_num_per_folder[i]
            inds_vis = torch.arange(int(self.start_ratio * num_img),
                                    int(self.end_ratio * num_img),
                                    frame_gap, dtype=int)
            if second_frame_range > 0:
                inds_lwir = inds_vis + torch.randint(-second_frame_range, second_frame_range, (inds_vis.shape[0],))
                inds_lwir[inds_lwir<int(self.start_ratio * num_img)] = int(self.start_ratio * num_img)
                inds_lwir[inds_lwir>int(self.end_ratio * num_img)-1] = int(self.end_ratio * num_img)-1
            else:
                inds_lwir = inds_vis
            for j, k in zip(inds_vis, inds_lwir):
                img_name0 = os.path.relpath(vis_img_paths[i][j], root_dir)
                img_name1 = os.path.relpath(lwir_img_paths[i][k], root_dir)

                if torch.rand(1) > 0.5:
                    img_name0, img_name1 = img_name1, img_name0

                pair_paths.append([img_name0, img_name1])
        
        random.shuffle(pair_paths)
        return pair_paths

    def __len__(self):
        return len(self.pair_paths)

    def __getitem__(self, idx):
        # read grayscale and normalized image, and mask. (1, h, w) and (h, w)
        img_name0 = osp.join(self.root_dir, self.pair_paths[idx][0])
        img_name1 = osp.join(self.root_dir, self.pair_paths[idx][1])

        if self.mode == "train" and torch.rand(1) > 0.5:
            img_name0, img_name1 = img_name1, img_name0

        image0, image0_norm, mask0, scale0, image0_mean, image0_std = read_pretrain_gray(
            img_name0, self.img_resize, self.df, self.img_padding, None)
        image1, image1_norm, mask1, scale1, image1_mean, image1_std = read_pretrain_gray(
            img_name1, self.img_resize, self.df, self.img_padding, None)

        data = {
            'image0': image0,  # (1, h, w)
            'image1': image1,
            'image0_norm': image0_norm,
            'image1_norm': image1_norm,
            'scale0': scale0,  # [scale_w, scale_h]
            'scale1': scale1,
            "image0_mean": image0_mean,
            "image0_std": image0_std,
            "image1_mean": image1_mean,
            "image1_std": image1_std,
            'dataset_name': 'PreTrain',
            'pair_id': idx,
            'pair_names': (self.pair_paths[idx][0], self.pair_paths[idx][1]),
        }

        # for XoFTR training
        if mask0 is not None:  # img_padding is True
            if self.coarse_scale:
                [ts_mask_0, ts_mask_1] = F.interpolate(torch.stack([mask0, mask1], dim=0)[None].float(),
                                                       scale_factor=self.coarse_scale,
                                                       mode='nearest',
                                                       recompute_scale_factor=False)[0].bool()
            data.update({'mask0': ts_mask_0, 'mask1': ts_mask_1})

        return data
