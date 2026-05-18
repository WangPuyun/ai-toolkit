import torch
from PIL import Image
from diffusers import Flux2KleinPipeline

pipe = Flux2KleinPipeline.from_pretrained(
    "/root/autodl-tmp/models/flux-klein-base-4b", torch_dtype=torch.bfloat16
)
pipe.load_lora_weights("/root/autodl-tmp/ai-toolkit/output/my_first_lora_v1/my_first_lora_v1_000002750.safetensors")
pipe.to("cuda")

# 加载图像
input_image = Image.open("/root/autodl-tmp/ai-toolkit/datasets/control1/0c7a92f4-adba-451a-b8df-f9159a9874d3.png").convert("RGB")
background_image = Image.open("/root/autodl-tmp/ai-toolkit/datasets/control2/0c7a92f4-adba-451a-b8df-f9159a9874d3.png").convert("RGB")

image = pipe(
    prompt="通过已存在的光影，重塑主体上的光照和阴影，实现自然融合的效果，避免突兀和错位。100%保留人物接触的物品。100%确保人物没有位移。人物自然融入场景，上方的柔和",  # Use your trigger word
    image=[input_image, background_image],
    num_inference_steps=50,
    guidance_scale=4.0,
).images[0]

image.save("test_lora.png")