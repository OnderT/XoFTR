from configs.data.base import cfg

cfg.DATASET.TRAIN_DATA_SOURCE = "KAIST"
cfg.DATASET.TRAIN_DATA_ROOT = "data/kaist-cvpr15"
cfg.DATASET.VAL_DATA_SOURCE = "KAIST"
cfg.DATASET.VAL_DATA_ROOT = cfg.DATASET.TEST_DATA_ROOT = "data/kaist-cvpr15"

cfg.DATASET.PRETRAIN_IMG_RESIZE = 640  
