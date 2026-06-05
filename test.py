import os
import glob
import torch
from PIL import Image
from diffusers import Flux2KleinPipeline

pipe = Flux2KleinPipeline.from_pretrained(
    "/root/autodl-tmp/models/flux-klein-base-4b", torch_dtype=torch.bfloat16
)
pipe.load_lora_weights("/root/autodl-tmp/ai-toolkit/output/Flux2_lora_v6_baseline/Flux2_lora_v6_baseline.safetensors")
pipe.to("cuda")

control1_dir = "/root/autodl-tmp/ai-toolkit/datasets/test_control1"
control2_dir = "/root/autodl-tmp/ai-toolkit/datasets/test_control2"
output_dir = "/root/autodl-tmp/ai-toolkit/datasets/test_output"
os.makedirs(output_dir, exist_ok=True)

txt_files = sorted(glob.glob(os.path.join(control1_dir, "*.txt")))

for txt_path in txt_files:
    stem = os.path.splitext(os.path.basename(txt_path))[0]
    input_image_path = os.path.join(control1_dir, f"{stem}.png")
    background_image_path = os.path.join(control2_dir, f"{stem}.png")
    output_path = os.path.join(output_dir, f"{stem}.png")

    with open(txt_path, "r", encoding="utf-8") as f:
        prompt = f.read().strip()

    print(f"处理 {stem} ...")

    input_image = Image.open(input_image_path).convert("RGB")
    background_image = Image.open(background_image_path).convert("RGB")

    image = pipe(
        prompt=prompt,
        image=[input_image, background_image],
        num_inference_steps=50,
        guidance_scale=4.0,
    ).images[0]

    image.save(output_path)
    print(f"已保存 {output_path}")
