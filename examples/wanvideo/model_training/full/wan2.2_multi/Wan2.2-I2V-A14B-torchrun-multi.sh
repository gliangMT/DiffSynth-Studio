#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <node_rank> <master_addr> <master_port> [timestamp]"
  echo "Example:"
  echo "  node0: $0 0 10.18.33.31 29501"
  echo "  node1: $0 1 10.18.33.31 29501"
  exit 1
fi

NODE_RANK="$1"
MASTER_ADDR="$2"
MASTER_PORT="$3"
TIMESTAMP="${4:-$(date +"%Y%m%d_%H%M%S")}"

NNODES=2
NPROC_PER_NODE=8

WORK_DIR="$(pwd)"
LOG_DIR="${WORK_DIR}/logs/S5000_train_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

export ATTENTION_IMPLEMENTATION="torch"
export TRAIN_WITH_TORCHRUN=1 # multi-node training with torchrun

# runtime env
export OMP_NUM_THREADS=4
export MUSA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MUSA_EXECUTION_TIMEOUT=32000000
export MCCL_PROTOS=2
export MCCL_ALGOS=1
export MCCL_BUFFSIZE=20971520
export MCCL_IB_GID_INDEX=3
export MCCL_NET_SHARED_BUFFERS=0
export MCCL_IB_TC=106
export MCCL_IB_QPS_PER_CONNECTION=16
export MCCL_IB_TIMEOUT=20
export MCCL_IB_RETRY_CNT=7
export MCCL_SOCKET_IFNAME=bond0
export MCCL_CROSS_NIC=0

echo "[INFO] NODE_RANK=${NODE_RANK}"
echo "[INFO] MASTER_ADDR=${MASTER_ADDR}"
echo "[INFO] MASTER_PORT=${MASTER_PORT}"
echo "[INFO] NNODES=${NNODES}"
echo "[INFO] NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "[INFO] MUSA_VISIBLE_DEVICES=${MUSA_VISIBLE_DEVICES}"
echo "[INFO] LOG_DIR=${LOG_DIR}"

torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  examples/wanvideo/model_training/train.py \
  --dataset_base_path /data/datasets/OpenVidHD/universal_datasets \
  --dataset_metadata_path /data/datasets/OpenVidHD/universal_datasets/metadata.csv \
  --height 480 \
  --width 832 \
  --num_frames 21 \
  --dataset_repeat 12 \
  --model_paths '[
      [
        "/data/liang.geng/shared_dir/DiffSynth-Studio/models/Wan-AI/Wan2.2-I2V-A14B/low_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors",
        "/data/liang.geng/shared_dir/DiffSynth-Studio/models/Wan-AI/Wan2.2-I2V-A14B/low_noise_model/diffusion_pytorch_model-00002-of-00006.safetensors",
        "/data/liang.geng/shared_dir/DiffSynth-Studio/models/Wan-AI/Wan2.2-I2V-A14B/low_noise_model/diffusion_pytorch_model-00003-of-00006.safetensors",
        "/data/liang.geng/shared_dir/DiffSynth-Studio/models/Wan-AI/Wan2.2-I2V-A14B/low_noise_model/diffusion_pytorch_model-00004-of-00006.safetensors",
        "/data/liang.geng/shared_dir/DiffSynth-Studio/models/Wan-AI/Wan2.2-I2V-A14B/low_noise_model/diffusion_pytorch_model-00005-of-00006.safetensors",
        "/data/liang.geng/shared_dir/DiffSynth-Studio/models/Wan-AI/Wan2.2-I2V-A14B/low_noise_model/diffusion_pytorch_model-00006-of-00006.safetensors"
      ],
      "/data/liang.geng/shared_dir/DiffSynth-Studio/models/Wan-AI/Wan2.2-I2V-A14B/models_t5_umt5-xxl-enc-bf16.pth",
      "/data/liang.geng/shared_dir/DiffSynth-Studio/models/Wan-AI/Wan2.2-I2V-A14B/Wan2.1_VAE.pth"
  ]' \
  --learning_rate 1e-5 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.2-I2V-A14B_low_noise_full_frames21_flash-attn-${NODE_RANK}" \
  --save_steps 500 \
  --trainable_models "dit" \
  --extra_inputs "input_image" \
  --use_gradient_checkpointing_offload \
  --max_timestep_boundary 1 \
  --min_timestep_boundary 0.358 \
  2>&1 | tee "${LOG_DIR}/train_${TIMESTAMP}_node${NODE_RANK}.log"