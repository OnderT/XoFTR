from collections import abc
from loguru import logger

import pytorch_lightning as pl
from torch import distributed as dist
from torch.utils.data import (
    DataLoader,
    ConcatDataset,
    DistributedSampler
)

from src.datasets.pretrain_dataset import PretrainDataset


class PretrainDataModule(pl.LightningDataModule):
    """ 
    For distributed training, each training process is assgined
    only a part of the training scenes to reduce memory overhead.
    """
    def __init__(self, args, config):
        super().__init__()

        # 1. data config
        # Train and Val should from the same data source
        self.train_data_source = config.DATASET.TRAIN_DATA_SOURCE
        self.val_data_source = config.DATASET.VAL_DATA_SOURCE
        # training and validating
        self.train_data_root = config.DATASET.TRAIN_DATA_ROOT
        self.val_data_root = config.DATASET.VAL_DATA_ROOT

        # 2. dataset config']

        # dataset options
        self.pretrain_img_resize = config.DATASET.PRETRAIN_IMG_RESIZE  # 840
        self.pretrain_img_pad = config.DATASET.PRETRAIN_IMG_PAD   # True
        self.pretrain_df = config.DATASET.PRETRAIN_DF  # 8
        self.coarse_scale = 1 / config.XOFTR.RESOLUTION[0]  # 0.125. for training xoftr.
        self.frame_gap = config.DATASET.PRETRAIN_FRAME_GAP

        # 3.loader parameters
        self.train_loader_params = {
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'pin_memory': getattr(args, 'pin_memory', True)
        }
        self.val_loader_params = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': args.num_workers,
            'pin_memory': getattr(args, 'pin_memory', True)
        }

    def setup(self, stage=None):
        """
        Setup train / val / test dataset. This method will be called by PL automatically.
        Args:
            stage (str): 'fit' in training phase, and 'test' in testing phase.
        """

        assert stage in ['fit', 'test'], "stage must be either fit or test"

        try:
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            logger.info(f"[rank:{self.rank}] world_size: {self.world_size}")
        except AssertionError as ae:
            self.world_size = 1
            self.rank = 0
            logger.warning(str(ae) + " (set wolrd_size=1 and rank=0)")

        if stage == 'fit':
            self.train_dataset = self._setup_dataset(
                self.train_data_root,
                mode='train')
            # setup multiple (optional) validation subsets
            self.val_dataset = []
            self.val_dataset.append(self._setup_dataset(
                self.val_data_root,
                mode='val'))
            logger.info(f'[rank:{self.rank}] Train & Val Dataset loaded!')
        else:  # stage == 'test
            raise ValueError(f"only 'fit' implemented")

    def _setup_dataset(self,
                       data_root,
                       mode='train'):
        """ Setup train / val / test set"""
        
        dataset_builder = self._build_concat_dataset
        return dataset_builder(data_root, mode=mode)

    def _build_concat_dataset(
        self,
        data_root,
        mode
    ):
        datasets = []

        datasets.append(
            PretrainDataset(data_root,
                                mode=mode,
                                img_resize=self.pretrain_img_resize,
                                df=self.pretrain_df,
                                img_padding=self.pretrain_img_pad,
                                coarse_scale=self.coarse_scale,
                                frame_gap=self.frame_gap))

        return ConcatDataset(datasets)

    def train_dataloader(self):
        """ Build training dataloader for KAIST dataset. """
        sampler = DistributedSampler(self.train_dataset, shuffle=True)
        dataloader = DataLoader(self.train_dataset, sampler=sampler, **self.train_loader_params)
        return dataloader
    
    def val_dataloader(self):
        """ Build validation dataloader KAIST dataset. """
        if not isinstance(self.val_dataset, abc.Sequence):
            return DataLoader(self.val_dataset, sampler=sampler, **self.val_loader_params)
        else:
            dataloaders = []
            for dataset in self.val_dataset:
                sampler = DistributedSampler(dataset, shuffle=False)
                dataloaders.append(DataLoader(dataset, sampler=sampler, **self.val_loader_params))
            return dataloaders
