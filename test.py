import os
import glob
import torch
from PIL import Image
from diffusers import Flux2KleinPipeline
import lpips

def image_to_tensor(image):
    """
    把 PIL 图像转成 LPIPS 需要的 tensor。
    LPIPS 要求数值范围在 [-1, 1]。
    """
    tensor = torch.tensor(list(image.getdata()), dtype=torch.float32)
    tensor = tensor.view(image.height, image.width, 3)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    tensor = tensor / 255.0
    tensor = tensor * 2.0 - 1.0
    return tensor

pipe = Flux2KleinPipeline.from_pretrained(
    "/root/autodl-tmp/models/flux-klein-base-4b", torch_dtype=torch.bfloat16
)
pipe.load_lora_weights("/root/autodl-tmp/ai-toolkit/output/Flux2_lora_v6_baseline/Flux2_lora_v6_baseline.safetensors")
pipe.to("cuda")

control1_dir = "/root/autodl-tmp/ai-toolkit/datasets/test_control1"
control2_dir = "/root/autodl-tmp/ai-toolkit/datasets/test_control2"
target_dir = "/root/autodl-tmp/ai-toolkit/datasets/test_target"
output_dir = "/root/autodl-tmp/ai-toolkit/datasets/test_output"
os.makedirs(output_dir, exist_ok=True)

txt_files = sorted(glob.glob(os.path.join(control1_dir, "*.txt")))

device = "cuda" if torch.cuda.is_available() else "cpu"
loss_fn = lpips.LPIPS(net="vgg").to(device)
loss_fn.eval()

total_lpips = 0.0

for txt_path in txt_files:
    stem = os.path.splitext(os.path.basename(txt_path))[0]
    input_image_path = os.path.join(control1_dir, f"{stem}.png")
    background_image_path = os.path.join(control2_dir, f"{stem}.png")
    target_image_path = os.path.join(target_dir, f"{stem}.png")
    output_path = os.path.join(output_dir, f"{stem}.png")

    with open(txt_path, "r", encoding="utf-8") as f:
        prompt = f.read().strip()

    print(f"处理 {stem} ...")

    input_image = Image.open(input_image_path).convert("RGB")
    background_image = Image.open(background_image_path).convert("RGB")
    target_image = Image.open(target_image_path).convert("RGB")

    image = pipe(
        prompt=prompt,
        image=[input_image, background_image],
        num_inference_steps=50,
        guidance_scale=4.0,
    ).images[0]

    image.save(output_path)
    print(f"已保存 {output_path}")

    target_image = target_image.resize(image.size, Image.LANCZOS)

    image_tensor = image_to_tensor(image).to(device)
    target_tensor = image_to_tensor(target_image).to(device)

    with torch.no_grad():
        score = loss_fn(image_tensor, target_tensor).item()

    total_lpips += score

print(f"平均 LPIPS: {total_lpips / len(txt_files):.4f}")

