import torch
from diffusers import Flux2KleinPipeline

pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-base-9B", torch_dtype=torch.bfloat16
)
pipe.load_lora_weights("path/to/your_lora.safetensors")
pipe.to("cuda")

image = pipe(
    "a photo of ohwx in a garden on a sunny day",  # Use your trigger word
    num_inference_steps=50,
    guidance_scale=4.0,
).images[0]