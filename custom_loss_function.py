import torch
import torch.nn as nn

_dwt = None

def _get_wavelet_loss(device, dtype):
    global _dwt
    if _dwt is not None:
        return _dwt

    # init wavelets
    from pytorch_wavelets import DWTForward

    # wave='db1'  wave='haar'
    dwt = DWTForward(J=1, mode="zero", wave="haar").to(device=device, dtype=dtype)
    _dwt = dwt
    return dwt

class CustomLoss(nn.Module):
    def __init__(self, mse_weight: float = 1.0, wave_weight: float = 1.0):
        super().__init__()
        self.mse_weight = float(mse_weight)
        self.wave_weight = float(wave_weight)

        # 逐元素子损失
        self.mse = nn.MSELoss(reduction="none")
        self.wave = nn.MSELoss(reduction="none")

        # 记录最近一次各项损失值，方便训练时打印观察
        self.last_terms = {}

    def forward(self, pred: torch.Tensor, target: torch.Tensor, latents: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        pred, target: [B, C, H, W] latent 张量（已被调用方转成 float32）。
        返回：与 pred 同形的【逐元素】损失。
        """
        assert pred.shape == target.shape, "pred 与 target 形状必须一致"

        loss_map = torch.zeros_like(pred)
        self.last_terms = {}

        if self.mse_weight > 0:
            term = self.mse(pred.float(), target.float())
            loss_map = loss_map + self.mse_weight * term
            self.last_terms["mse"] = term.detach().mean().item()

        if self.wave_weight > 0:
            model_pred = pred.float()
            latents = latents.float()
            noise = noise.float()
            dwt = _get_wavelet_loss(model_pred.device, model_pred.dtype)
            with torch.no_grad():
                model_input_xll, model_input_xh = dwt(latents)
                model_input_xlh, model_input_xhl, model_input_xhh = torch.unbind(
                    model_input_xh[0], dim=2
                )
                model_input = torch.cat(
                    [model_input_xll, model_input_xlh, model_input_xhl, model_input_xhh], dim=1
                )

            # reverse the noise to get the model prediction of the pure latents
            model_pred = noise - model_pred

            model_pred_xll, model_pred_xh = dwt(model_pred)
            model_pred_xlh, model_pred_xhl, model_pred_xhh = torch.unbind(
                model_pred_xh[0], dim=2
            )
            model_pred = torch.cat(
                [model_pred_xll, model_pred_xlh, model_pred_xhl, model_pred_xhh], dim=1
            )

            # wave 支路经 DWT 后形状为 [B, 4C, H/2, W/2]，与 loss_map 的 [B, C, H, W] 不兼容，
            # 无法直接逐元素相加。这里先 mean 成标量，再作为常数偏置广播到 loss_map 上。
            # SDTrainer 下游 loss.mean([1,2,3]) 对常数偏置是保留的（不除 C·H·W），所以
            # wave_weight 仍按“平均小波损失”的权重语义生效。
            term = self.wave(model_pred, model_input).mean()

            loss_map = loss_map + self.wave_weight * term
            self.last_terms["wave"] = term.detach().item()
            print(self.last_terms["wave"])

        return loss_map


# --------------------------------------------------------------------------- #
# 简单自检：python custom_loss_function.py
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, H, W = 2, 16, 64, 64
    pred = torch.randn(B, C, H, W)
    target = torch.randn(B, C, H, W)

    crit = CustomLoss(mse_weight=1.0, wave_weight=1.0)
    out = crit(pred, target)
    print("loss shape:", tuple(out.shape), "(应与 pred 同形)")
    print("各项子损失:", crit.last_terms)
    print("最终标量:", out.mean().item())
