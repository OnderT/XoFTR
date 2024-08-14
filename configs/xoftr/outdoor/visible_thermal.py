from src.config.default import _CN as cfg

cfg.XOFTR.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'

cfg.TRAINER.CANONICAL_LR = 8e-3
cfg.TRAINER.WARMUP_STEP = 1875  # 3 epochs
cfg.TRAINER.WARMUP_RATIO = 0.1
cfg.TRAINER.MSLR_MILESTONES = [8, 12, 16, 20, 24, 30, 36, 42]

# pose estimation
cfg.TRAINER.RANSAC_PIXEL_THR = 1.5

cfg.TRAINER.OPTIMIZER = "adamw"
cfg.TRAINER.ADAMW_DECAY = 0.1
cfg.XOFTR.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.3

cfg.TRAINER.USE_WANDB = True # use weight and biases
