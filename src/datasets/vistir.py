import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from loguru import logger

from src.utils.dataset import read_vistir_gray

class VisTirDataset(Dataset):
    def __init__(self,
                 root_dir,
                 npz_path,
                 mode='val',
                 img_resize=None,
                 df=None,
                 img_padding=False,
                 **kwargs):
        """
        Manage one scene(npz_path) of VisTir dataset.
        
        Args:
            root_dir (str): VisTIR root directory.
            npz_path (str): {scene_id}.npz path. This contains image pair information of a scene.
            mode (str): options are ['val', 'test']
            img_resize (int, optional): the longer edge of resized images. None for no resize. 640 is recommended.
            df (int, optional): image size division factor. NOTE: this will change the final image size after img_resize.
            img_padding (bool): If set to 'True', zero-pad the image to squared size.
        """
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.scene_id = npz_path.split('.')[0]

        # prepare scene_info and pair_info
        self.scene_info = dict(np.load(npz_path, allow_pickle=True))
        self.pair_infos = self.scene_info['pair_infos'].copy()
        del self.scene_info['pair_infos']

        # parameters for image resizing, padding 
        self.img_resize = img_resize
        self.df = df
        self.img_padding = img_padding

        # for training XoFTR
        self.coarse_scale = getattr(kwargs, 'coarse_scale', 0.125)


    def __len__(self):
        return len(self.pair_infos)

    def __getitem__(self, idx):
        (idx0, idx1) = self.pair_infos[idx]

        
        img_name0 = osp.join(self.root_dir, self.scene_info['image_paths'][idx0][0])
        img_name1 = osp.join(self.root_dir, self.scene_info['image_paths'][idx1][1])

        # read intrinsics of original size
        K_0 = np.array(self.scene_info['intrinsics'][idx0][0], dtype=float).reshape(3,3)
        K_1 = np.array(self.scene_info['intrinsics'][idx1][1], dtype=float).reshape(3,3)

        # read distortion coefficients 
        dist0 =  np.array(self.scene_info['distortion_coefs'][idx0][0], dtype=float)
        dist1 = np.array(self.scene_info['distortion_coefs'][idx1][1], dtype=float)

        # read grayscale undistorted image and mask. (1, h, w) and (h, w)
        image0, mask0, scale0, K_0 = read_vistir_gray(
            img_name0, K_0, dist0, self.img_resize, self.df, self.img_padding, augment_fn=None)
        image1, mask1, scale1, K_1 = read_vistir_gray(
            img_name1, K_1, dist1, self.img_resize, self.df, self.img_padding, augment_fn=None)

        # to tensor
        K_0 = torch.tensor(K_0.copy(), dtype=torch.float).reshape(3, 3)
        K_1 = torch.tensor(K_1.copy(), dtype=torch.float).reshape(3, 3)

        # read and compute relative poses
        T0 = self.scene_info['poses'][idx0]
        T1 = self.scene_info['poses'][idx1]
        T_0to1 = torch.tensor(np.matmul(T1, np.linalg.inv(T0)), dtype=torch.float)[:4, :4]  # (4, 4)
        T_1to0 = T_0to1.inverse()

        data = {
            'image0': image0,  # (1, h, w)
            'image1': image1,
            'T_0to1': T_0to1,  # (4, 4)
            'T_1to0': T_1to0,
            'K0': K_0,  # (3, 3)
            'K1': K_1,
            'dist0': dist0,
            'dist1': dist1,
            'scale0': scale0,  # [scale_w, scale_h]
            'scale1': scale1,
            'dataset_name': 'VisTir',
            'scene_id': self.scene_id,
            'pair_id': idx,
            'pair_names': (self.scene_info['image_paths'][idx0][0], self.scene_info['image_paths'][idx1][1]),
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
