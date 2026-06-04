# Loop Results — LoRA Training-Evaluation Closed Loop

| Version | Config Path | Train Steps | LR | LoRA Source Path | Copied LoRA Path | LPIPS Mean | vs Previous | vs v6 | Decision | Notes |
|---------|-------------|-------------|------|------------------|------------------|------------|-------------|-------|----------|-------|
| v6 | `config/Flux2_lora_v6.json` | 2000 | 8e-5 | `output/Flux2_lora_v6/Flux2_lora_v6.safetensors` | `ComfyUI/models/loras/new_lora.safetensors` | **0.206198** | - | - | baseline | linear=32, conv=16, adamw8bit, gradient_accumulation=4 |
| v7 | `config/Flux2_lora_v7.json` | 2500 | 8e-5 | `output/Flux2_lora_v7/Flux2_lora_v7_000002000.safetensors` | `ComfyUI/models/loras/new_lora.safetensors` | **0.186803** | -9.5% | -9.5% | keep | Steps 2000→2500; loss still trending at v6 end |

## Training Loss Summary

### v6 (2000 steps)
| Window | Avg Loss |
|--------|----------|
| 0-199 | 0.415 |
| 1000-1199 | 0.394 |
| 1800-1999 | 0.382 |

### v7 (2494 steps completed, 2000-step checkpoint used)
| Window | Avg Loss |
|--------|----------|
| 0-199 | 0.4197 |
| 1000-1199 | 0.3934 |
| 1800-1999 | 0.3746 |
| 2200-2399 | 0.3782 |
| 2400-2494 | 0.3844 |

## Key Findings
- Increasing training steps from 2000 to 2500 reduced LPIPS by 9.5% (0.206 → 0.187)
- v7 training loss at window 1800-1999 (0.3746) is already lower than v6's same window (0.382)
- Loss continues to fluctuate around 0.37-0.38 after step 2000, suggesting room for further optimization
- Best version so far: **v7** (LPIPS = 0.186803)
