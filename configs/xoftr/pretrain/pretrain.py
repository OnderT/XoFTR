from src.config.default import _CN as cfg

cfg.TRAINER.CANONICAL_LR = 4e-3
cfg.TRAINER.WARMUP_STEP = 1250  # 2 epochs
cfg.TRAINER.WARMUP_RATIO = 0.1
cfg.TRAINER.MSLR_MILESTONES = [4, 6, 8, 10, 12, 14, 16, 18]

cfg.TRAINER.OPTIMIZER = "adamw"
cfg.TRAINER.ADAMW_DECAY = 0.1

cfg.TRAINER.USE_WANDB = True # use weight and biases

