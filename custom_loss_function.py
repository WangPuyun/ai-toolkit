import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self, mse_weight: float = 1.0, l1_weight: float = 1.0):
        super().__init__()
        self.mse_weight = float(mse_weight)
        self.l1_weight = float(l1_weight)

        # 逐元素子损失
        self.mse = nn.MSELoss(reduction="none")
        self.l1 = nn.L1Loss(reduction="none")

        # 记录最近一次各项损失值，方便训练时打印观察
        self.last_terms = {}

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target: [B, C, H, W] latent 张量（已被调用方转成 float32）。
        返回：与 pred 同形的【逐元素】损失。
        """
        assert pred.shape == target.shape, "pred 与 target 形状必须一致"

        loss_map = torch.zeros_like(pred)
        self.last_terms = {}

        if self.mse_weight > 0:
            term = self.mse(pred, target)
            loss_map = loss_map + self.mse_weight * term
            self.last_terms["mse"] = term.detach().mean().item()

        if self.l1_weight > 0:
            term = self.l1(pred, target)
            loss_map = loss_map + self.l1_weight * term
            self.last_terms["l1"] = term.detach().mean().item()

        return loss_map


# --------------------------------------------------------------------------- #
# 简单自检：python custom_loss_function.py
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, H, W = 2, 16, 64, 64
    pred = torch.randn(B, C, H, W)
    target = torch.randn(B, C, H, W)

    crit = CustomLoss(mse_weight=1.0, l1_weight=1.0)
    out = crit(pred, target)
    print("loss shape:", tuple(out.shape), "(应与 pred 同形)")
    print("各项子损失:", crit.last_terms)
    print("最终标量:", out.mean().item())
