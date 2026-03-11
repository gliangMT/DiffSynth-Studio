WORK_DIR=$(pwd) # Set the working directory to the current directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR=${WORK_DIR}/logs/S5000_train_${TIMESTAMP}
mkdir -p "${LOG_DIR}"

export ATTENTION_IMPLEMENTATION= "torch"
# export MUSA_LAUNCH_BLOCKING=1 # Enable blocking mode for better stability on MUSA
# echo "[DEBUG] MUSA_LAUNCH_BLOCKING is set to ${MUSA_LAUNCH_BLOCKING}. This may help with stability on MUSA devices."

export MASTER_ADDR=10.18.33.31
export MASTER_PORT=29510

# MCCL ENVRIONMENT VARIABLES
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

accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B_host0.yaml examples/wanvideo/model_training/train.py \
  --dataset_base_path /data/datasets/OpenVidHD/universal_datasets \
  --dataset_metadata_path /data/datasets/OpenVidHD/universal_datasets/metadata.csv \
  --height 480 \
  --width 832 \
  --num_frames 121 \
  --dataset_repeat 3 \
  --model_paths '[
      "models/train/H200_safetensors_flashattn/step-100.safetensors",
      "models/Wan-AI/Wan2.2-I2V-A14B/models_t5_umt5-xxl-enc-bf16.pth",
      "models/Wan-AI/Wan2.2-I2V-A14B/Wan2.1_VAE.pth"
  ]' \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.2-I2V-A14B_low_noise_full_frames121_flash-attn" \
  --save_steps 50 \
  --trainable_models "dit" \
  --extra_inputs "input_image" \
  --use_gradient_checkpointing_offload \
  --max_timestep_boundary 1 \
  --min_timestep_boundary 0.358 2>&1 | tee "${LOG_DIR}/train_${TIMESTAMP}_low_noise.log"
# boundary corresponds to timesteps [0, 900)

# --model_id_with_origin_paths "Wan-AI/Wan2.2-I2V-A14B:low_noise_model/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-I2V-A14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-I2V-A14B:Wan2.1_VAE.pth" \

  # --model_paths '[
  #     [
  #         "models/Wan-AI/Wan2.2-I2V-A14B/low_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors",
  #         "models/Wan-AI/Wan2.2-I2V-A14B/low_noise_model/diffusion_pytorch_model-00002-of-00006.safetensors",
  #         "models/Wan-AI/Wan2.2-I2V-A14B/low_noise_model/diffusion_pytorch_model-00003-of-00006.safetensors",
  #         "models/Wan-AI/Wan2.2-I2V-A14B/low_noise_model/diffusion_pytorch_model-00004-of-00006.safetensors",
  #         "models/Wan-AI/Wan2.2-I2V-A14B/low_noise_model/diffusion_pytorch_model-00005-of-00006.safetensors",
  #         "models/Wan-AI/Wan2.2-I2V-A14B/low_noise_model/diffusion_pytorch_model-00006-of-00006.safetensors",
  #     ],
  #     "models/Wan-AI/Wan2.2-I2V-A14B/models_t5_umt5-xxl-enc-bf16.pth",
  #     "models/Wan-AI/Wan2.2-I2V-A14B/Wan2.1_VAE.pth"
  # ]' \

# load ckpt from step N for continuing training
  # --model_paths '[
  #     "models/train/Wan2.2-I2V-A14B_low_noise_full_frames121_flash-attn/step-50.safetensors",
  #     "models/Wan-AI/Wan2.2-I2V-A14B/models_t5_umt5-xxl-enc-bf16.pth",
  #     "models/Wan-AI/Wan2.2-I2V-A14B/Wan2.1_VAE.pth"
  # ]' \