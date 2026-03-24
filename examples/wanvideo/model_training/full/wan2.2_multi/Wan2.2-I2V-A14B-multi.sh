#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <config_file> <node_tag> <master_addr> <hostfile> [master_port] [timestamp]"
  echo "Example:"
  echo "  $0 examples/wanvideo/model_training/full/accelerate_config_14B_host0.yaml node0 10.18.33.31 29510"
  exit 1
fi

CONFIG_FILE="$1"
NODE_TAG="$2"
MASTER_ADDR="$3"
HOSTFILE="$4"
MASTER_PORT="${5:-29501}"
TIMESTAMP="${6:-$(date +"%Y%m%d_%H%M%S")}"

WORK_DIR="$(pwd)"
LOG_DIR="${WORK_DIR}/logs/S5000_train_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

# basic env
export ATTENTION_IMPLEMENTATION="torch"
# export MUSA_LAUNCH_BLOCKING=1

# rendezvous env
export MASTER_ADDR="${MASTER_ADDR}"
export MASTER_PORT="${MASTER_PORT}"

# MCCL / runtime env
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

echo "[INFO] NODE_TAG=${NODE_TAG}"
echo "[INFO] CONFIG_FILE=${CONFIG_FILE}"
echo "[INFO] MASTER_ADDR=${MASTER_ADDR}"
echo "[INFO] MASTER_PORT=${MASTER_PORT}"
echo "[INFO] HOSTFILE=${HOSTFILE}"
echo "[INFO] LOG_DIR=${LOG_DIR}"
echo "[INFO] MUSA_VISIBLE_DEVICES=${MUSA_VISIBLE_DEVICES}"

accelerate launch \
  --config_file "${CONFIG_FILE}" \
  --num_machines 2 \
  --num_processes 16 \
  --machine_rank "${NODE_TAG}" \
  --main_process_ip "${MASTER_ADDR}" \
  --main_process_port "${MASTER_PORT}" \
  --rdzv_backend static \
  --same_network \
  --deepspeed_hostfile "${HOSTFILE}" \
  examples/wanvideo/model_training/train.py \
  --dataset_base_path /data/datasets/OpenVidHD/universal_datasets \
  --dataset_metadata_path /data/datasets/OpenVidHD/universal_datasets/metadata.csv \
  --height 480 \
  --width 832 \
  --num_frames 21 \
  --dataset_repeat 3 \
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
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.2-I2V-A14B_low_noise_full_frames21_flash-attn-${NODE_TAG}" \
  --save_steps 500 \
  --trainable_models "dit" \
  --extra_inputs "input_image" \
  --use_gradient_checkpointing_offload \
  --max_timestep_boundary 1 \
  --min_timestep_boundary 0.358 \
  2>&1 | tee "${LOG_DIR}/train_${TIMESTAMP}_low_noise_${NODE_TAG}.log"