# Configuration Differences

## v6 → v7

| Field | v6 | v7 | Reason | Expected Effect |
|-------|----|----|--------|----------------|
| `name` | Flux2_lora_v6 | Flux2_lora_v7 | Version identifier | Distinguish output |
| `train.steps` | 2000 | 2500 | v6 loss still trending down at step 2000 (0.415→0.382), indicating room for further optimization | Lower training loss, potentially lower LPIPS |

All other parameters remain identical.

## v7 → v8

| Field | v7 | v8 | Reason | Expected Effect |
|-------|----|----|--------|----------------|
| `name` | Flux2_lora_v7 | Flux2_lora_v8 | Version identifier | Distinguish output |
| `train.steps` | 2500 | 3000 | v7 loss still fluctuating around 0.37-0.38 at steps 2200-2494; increasing steps gave +9.5% LPIPS improvement in v6→v7 | Continue loss reduction, potentially lower LPIPS |

All other parameters remain identical.

## v8 → v9 (pending)

To be filled after v8 evaluation.
