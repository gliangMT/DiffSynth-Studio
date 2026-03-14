import torchada
import torch
from PIL import Image
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="musa",
    # model_configs=[
    #     ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors"),
    #     ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors"),
    #     ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
    #     ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="Wan2.1_VAE.pth"),
    # ],
    model_configs=[
        ModelConfig(path=[
            "/data/liang.geng/DiffSynth-Studio/models/Wan-AI/Wan2.2-I2V-A14B/high_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors",
            "/data/liang.geng/DiffSynth-Studio/models/Wan-AI/Wan2.2-I2V-A14B/high_noise_model/diffusion_pytorch_model-00002-of-00006.safetensors",
            "/data/liang.geng/DiffSynth-Studio/models/Wan-AI/Wan2.2-I2V-A14B/high_noise_model/diffusion_pytorch_model-00003-of-00006.safetensors",
            "/data/liang.geng/DiffSynth-Studio/models/Wan-AI/Wan2.2-I2V-A14B/high_noise_model/diffusion_pytorch_model-00004-of-00006.safetensors",
            "/data/liang.geng/DiffSynth-Studio/models/Wan-AI/Wan2.2-I2V-A14B/high_noise_model/diffusion_pytorch_model-00005-of-00006.safetensors",
            "/data/liang.geng/DiffSynth-Studio/models/Wan-AI/Wan2.2-I2V-A14B/high_noise_model/diffusion_pytorch_model-00006-of-00006.safetensors",
        ]),
        ModelConfig(path="/data/liang.geng/DiffSynth-Studio/models/train/Wan2.2-I2V-A14B_low_noise_full_frames121_flash-attn/step-150.safetensors"),
        ModelConfig(path="/data/liang.geng/DiffSynth-Studio/models/Wan-AI/Wan2.2-I2V-A14B/models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(path="/data/liang.geng/DiffSynth-Studio/models/Wan-AI/Wan2.2-I2V-A14B/Wan2.1_VAE.pth"),
    ],
    tokenizer_config=ModelConfig(path="/data/liang.geng/DiffSynth-Studio/models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/"),
)

dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/examples_in_diffsynth",
    local_dir="./",
    allow_file_pattern=["data/examples/wan/cat_fightning.jpg"]
)
input_image = Image.open("data/examples/wan/cat_fightning.jpg").resize((832, 480))

video = pipe(
    prompt="Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=0, tiled=True,
    input_image=input_image,
    switch_DiT_boundary=0.9,
)
save_video(video, "21frame_step150_flash_attn_video_Wan2.2-I2V-A14B.mp4", fps=15, quality=5)
