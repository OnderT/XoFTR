#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"

# conda activate xoftr
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

TRAIN_IMG_SIZE=640
# TRAIN_IMG_SIZE=840
data_cfg_path="configs/data/megadepth_vistir_trainval_${TRAIN_IMG_SIZE}.py"
main_cfg_path="configs/xoftr/outdoor/visible_thermal.py"

n_nodes=1
n_gpus_per_node=8
torch_num_workers=16
batch_size=2
pin_memory=true
exp_name="visible_thermal-${TRAIN_IMG_SIZE}-bs=$(($n_gpus_per_node * $n_nodes * $batch_size))"
ckpt_path="pretrain_weights/epoch=8-.ckpt"

python -u ./train.py \
    ${data_cfg_path} \
    ${main_cfg_path} \
    --exp_name=${exp_name} \
    --gpus=${n_gpus_per_node} --num_nodes=${n_nodes} --accelerator="ddp" \
    --batch_size=${batch_size} --num_workers=${torch_num_workers} --pin_memory=${pin_memory} \
    --check_val_every_n_epoch=1 \
    --log_every_n_steps=100 \
    --limit_val_batches=1. \
    --num_sanity_val_steps=10 \
    --benchmark=True \
    --max_epochs=30 \
    --ckpt_path=${ckpt_path}
