
from loguru import logger

import torch
import pytorch_lightning as pl
from matplotlib import pyplot as plt
plt.switch_backend('agg')

from src.xoftr import XoFTR_Pretrain
from src.losses.xoftr_loss_pretrain import XoFTRLossPretrain
from src.optimizers import build_optimizer, build_scheduler
from src.utils.plotting import make_mae_figures
from src.utils.comm import all_gather
from src.utils.misc import lower_config, flattenList
from src.utils.profiler import PassThroughProfiler
from src.utils.pretrain_utils import generate_random_masks, get_target


class PL_XoFTR_Pretrain(pl.LightningModule):
    def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None):
        """
        TODO:
            - use the new version of PL logging API.
        """
        super().__init__()
        # Misc
        self.config = config  # full config
        
        _config = lower_config(self.config)
        self.xoftr_cfg = lower_config(_config['xoftr'])
        self.profiler = profiler or PassThroughProfiler()
        self.n_vals_plot = max(config.TRAINER.N_VAL_PAIRS_TO_PLOT // config.TRAINER.WORLD_SIZE, 1)

        # generator to create the same masks for validation
        self.val_seed = self.config.PRETRAIN.VAL_SEED
        self.val_generator = torch.Generator(device="cuda").manual_seed(self.val_seed)
        self.mae_margins = config.PRETRAIN.MAE_MARGINS

        # Matcher: XoFTR
        self.matcher = XoFTR_Pretrain(config=_config['xoftr'])
        self.loss = XoFTRLossPretrain(_config)

        # Pretrained weights
        if pretrained_ckpt:
            state_dict = torch.load(pretrained_ckpt, map_location='cpu')['state_dict']
            self.matcher.load_state_dict(state_dict, strict=False)
            logger.info(f"Load \'{pretrained_ckpt}\' as pretrained checkpoint")
        
        # Testing
        self.dump_dir = dump_dir
        
    def configure_optimizers(self):
        # FIXME: The scheduler did not work properly when `--resume_from_checkpoint`
        optimizer = build_optimizer(self, self.config)
        scheduler = build_scheduler(self.config, optimizer)
        return [optimizer], [scheduler]
    
    def optimizer_step(
            self, epoch, batch_idx, optimizer, optimizer_idx,
            optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        # learning rate warm up
        warmup_step = self.config.TRAINER.WARMUP_STEP
        if self.trainer.global_step < warmup_step:
            if self.config.TRAINER.WARMUP_TYPE == 'linear':
                base_lr = self.config.TRAINER.WARMUP_RATIO * self.config.TRAINER.TRUE_LR
                lr = base_lr + \
                    (self.trainer.global_step / self.config.TRAINER.WARMUP_STEP) * \
                    abs(self.config.TRAINER.TRUE_LR - base_lr)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
            elif self.config.TRAINER.WARMUP_TYPE == 'constant':
                pass
            else:
                raise ValueError(f'Unknown lr warm-up strategy: {self.config.TRAINER.WARMUP_TYPE}')

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
    
    def _trainval_inference(self, batch, generator=None):
        generate_random_masks(batch,
                        patch_size=self.config.PRETRAIN.PATCH_SIZE,
                        mask_ratio=self.config.PRETRAIN.MASK_RATIO,
                        generator=generator,
                        margins=self.mae_margins)
        
        with self.profiler.profile("XoFTR"):
            self.matcher(batch)
            
        with self.profiler.profile("Compute losses"):
            # Create target pacthes to reconstruct
            get_target(batch)
            self.loss(batch)
    
    def training_step(self, batch, batch_idx):
        self._trainval_inference(batch)
        
        # logging
        if self.trainer.global_rank == 0 and self.global_step % self.trainer.log_every_n_steps == 0:
            # scalars
            for k, v in batch['loss_scalars'].items():
                self.logger[0].experiment.add_scalar(f'train/{k}', v, self.global_step)
                if self.config.TRAINER.USE_WANDB:
                    self.logger[1].log_metrics({f'train/{k}': v}, self.global_step)
            
            if self.config.TRAINER.ENABLE_PLOTTING:
                figures = make_mae_figures(batch)
                for i, figure in enumerate(figures):
                    self.logger[0].experiment.add_figure(
                    f'train_mae/node_{self.trainer.global_rank}-device_{self.device.index}-batch_{i}',
                    figure, self.global_step)

        return {'loss': batch['loss']}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        if self.trainer.global_rank == 0:
            self.logger[0].experiment.add_scalar(
                'train/avg_loss_on_epoch', avg_loss,
                global_step=self.current_epoch)
            if self.config.TRAINER.USE_WANDB:
                self.logger[1].log_metrics(
                    {'train/avg_loss_on_epoch': avg_loss},
                    self.current_epoch)
    
    def validation_step(self, batch, batch_idx):
        self._trainval_inference(batch, self.val_generator)
    
        val_plot_interval = max(self.trainer.num_val_batches[0] // \
                                 (self.trainer.num_gpus * self.n_vals_plot), 1)
        figures = []
        if batch_idx % val_plot_interval == 0:
            figures = make_mae_figures(batch)

        return {
            'loss_scalars': batch['loss_scalars'],
            'figures': figures,
        }
        
    def validation_epoch_end(self, outputs):
        self.val_generator.manual_seed(self.val_seed)
        # handle multiple validation sets
        multi_outputs = [outputs] if not isinstance(outputs[0], (list, tuple)) else outputs
        
        for valset_idx, outputs in enumerate(multi_outputs):
            # since pl performs sanity_check at the very begining of the training
            cur_epoch = self.trainer.current_epoch
            if not self.trainer.resume_from_checkpoint and self.trainer.running_sanity_check:
                cur_epoch = -1

            # 1. loss_scalars: dict of list, on cpu
            _loss_scalars = [o['loss_scalars'] for o in outputs]
            loss_scalars = {k: flattenList(all_gather([_ls[k] for _ls in _loss_scalars])) for k in _loss_scalars[0]}
            
            _figures = [o['figures'] for o in outputs]
            figures = [item for sublist in _figures for item in sublist]

            # tensorboard records only on rank 0
            if self.trainer.global_rank == 0:
                for k, v in loss_scalars.items():
                    mean_v = torch.stack(v).mean()
                    self.logger[0].experiment.add_scalar(f'val_{valset_idx}/avg_{k}', mean_v, global_step=cur_epoch)
                    if self.config.TRAINER.USE_WANDB:
                        self.logger[1].log_metrics({f'val_{valset_idx}/avg_{k}': mean_v}, cur_epoch)

                for plot_idx, fig in enumerate(figures):
                    self.logger[0].experiment.add_figure(
                        f'val_mae_{valset_idx}/pair-{plot_idx}', fig, cur_epoch, close=True)

            plt.close('all')

