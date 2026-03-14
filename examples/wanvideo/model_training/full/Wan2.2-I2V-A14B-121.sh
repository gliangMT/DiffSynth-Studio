WORK_DIR=$(pwd) # Set the working directory to the current directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR=${WORK_DIR}/logs/S5000_train_frames-121-${TIMESTAMP}
mkdir -p "${LOG_DIR}"

export ATTENTION_IMPLEMENTATION = "torch"
# export MUSA_LAUNCH_BLOCKING=1 # Enable blocking mode for better stability on MUSA
# echo "[DEBUG] MUSA_LAUNCH_BLOCKING is set to ${MUSA_LAUNCH_BLOCKING}. This may help with stability on MUSA devices."

export MUSA_FLASH_ATTENTION_DEBUG=1 # Enable blocking mode for better stability on MUSA

accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B_test.yaml examples/wanvideo/model_training/train.py \
  --dataset_base_path /data/datasets/OpenVidHD/universal_datasets \
  --dataset_metadata_path /data/datasets/OpenVidHD/universal_datasets/metadata.csv \
  --height 480 \
  --width 832 \
  --num_frames 121 \
  --dataset_repeat 3 \
  --model_paths '[
    "/data/liang.geng/DiffSynth-Studio/models/train/Wan2.2-I2V-A14B_low_noise_full_frames121_flash-attn/step50_with_nan_step_66.safetensors",
    "/data/liang.geng/DiffSynth-Studio/models/Wan-AI/Wan2.2-I2V-A14B/models_t5_umt5-xxl-enc-bf16.pth",
    "/data/liang.geng/DiffSynth-Studio/models/Wan-AI/Wan2.2-I2V-A14B/Wan2.1_VAE.pth"
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
  #   [
  #     "/data/liang.geng/DiffSynth-Studio/models/Wan-AI/Wan2.2-I2V-A14B/low_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors",
  #     "/data/liang.geng/DiffSynth-Studio/models/Wan-AI/Wan2.2-I2V-A14B/low_noise_model/diffusion_pytorch_model-00002-of-00006.safetensors",
  #     "/data/liang.geng/DiffSynth-Studio/models/Wan-AI/Wan2.2-I2V-A14B/low_noise_model/diffusion_pytorch_model-00003-of-00006.safetensors",
  #     "/data/liang.geng/DiffSynth-Studio/models/Wan-AI/Wan2.2-I2V-A14B/low_noise_model/diffusion_pytorch_model-00004-of-00006.safetensors",
  #     "/data/liang.geng/DiffSynth-Studio/models/Wan-AI/Wan2.2-I2V-A14B/low_noise_model/diffusion_pytorch_model-00005-of-00006.safetensors",
  #     "/data/liang.geng/DiffSynth-Studio/models/Wan-AI/Wan2.2-I2V-A14B/low_noise_model/diffusion_pytorch_model-00006-of-00006.safetensors"
  #   ],
  #   "/data/liang.geng/DiffSynth-Studio/models/Wan-AI/Wan2.2-I2V-A14B/models_t5_umt5-xxl-enc-bf16.pth",
  #   "/data/liang.geng/DiffSynth-Studio/models/Wan-AI/Wan2.2-I2V-A14B/Wan2.1_VAE.pth"
  # ]' \

# load ckpt from step N for continuing training, the ckpt may cause nan issue, for debugging purpose
  # --model_paths '[
  #   "/data/liang.geng/DiffSynth-Studio/models/train/Wan2.2-I2V-A14B_low_noise_full_frames121_flash-attn/step-100.safetensors",
  #   "/data/liang.geng/DiffSynth-Studio/models/Wan-AI/Wan2.2-I2V-A14B/models_t5_umt5-xxl-enc-bf16.pth",
  #   "/data/liang.geng/DiffSynth-Studio/models/Wan-AI/Wan2.2-I2V-A14B/Wan2.1_VAE.pth"
  # ]' \