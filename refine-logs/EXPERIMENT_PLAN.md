# 实验计划

> **Workflow 1.5 (`/experiment-bridge`) 的模板。** 填写后保存为 `refine-logs/EXPERIMENT_PLAN.md`，然后运行 `/experiment-bridge`。

**问题**: 当前 Flux2 LoRA 训练依赖人工调参，训练完成后还需要手动复制权重、切换到 ComfyUI 评估脚本、统计测试集 LPIPS 均值，再根据结果修改下一版训练配置。目标是将该流程交给 ARIS 自动完成，形成可复现的“训练—评估—调参—再训练”闭环。

**方法论点**: 以 `/root/autodl-tmp/ai-toolkit/config/Flux2_lora_v6.json` 为初始配置，使用 AI-Toolkit 训练 LoRA；将每轮训练得到的 `Flux2_lora_v*.safetensors` 复制为 ComfyUI 使用的 `new_lora.safetensors`；运行 `comfy_api_lora_lpips_eval.py` 计算测试集 LPIPS 均值；基于 LPIPS 均值迭代生成 `Flux2_lora_v7.json`、`Flux2_lora_v8.json`、`Flux2_lora_v9.json` 等后续配置，目标是逐步降低测试集 LPIPS 均值。

## Claim 映射

| Claim | 重要性 | 最低说服力证据 | 关联实验块 |
|-------|--------|---------------|-----------|
| C1: 自动闭环调参可以找到 LPIPS 均值更低的 LoRA 配置 | 证明 ARIS 能减少人工调参成本，并以客观指标驱动配置优化 | 至少完成 v6 → v7 → v8 的连续实验记录；后续版本 LPIPS 均值低于 v6，或能解释未降低的原因 | B1, B2 |
| C2: 每轮实验结果是可追踪、可复现的 | 防止只得到一个最终 LoRA，却无法追溯参数、权重、指标和日志 | 每轮保存 JSON 配置、训练日志、LoRA 权重路径、评估输出、LPIPS 均值、修改理由 | B1, B3 |
| C3: 参数修改来自评估结果，而非随机大幅改动 | 降低无效搜索和训练资源浪费 | 每个新版本配置文件必须说明相对上一版本修改了哪些字段，以及修改原因 | B2, B3 |

## 实验块

### 实验块 1: 主实验——LoRA 训练与 LPIPS 评估闭环
- **验证 Claim**: C1, C2
- **数据集 / 划分 / 任务**:
  - 使用当前 AI-Toolkit 配置文件 `Flux2_lora_v6.json` 中已定义的训练集、验证/采样设置、模型路径和输出路径。
  - 使用 `/root/autodl-tmp/ComfyUI/comfy_api_lora_lpips_eval.py` 中已定义的测试集进行评估。
  - 任务为 Flux2 LoRA 训练质量评估，主要指标为整个测试集上的 LPIPS 均值。
- **对比系统**:
  - 起点配置：`Flux2_lora_v6.json`
  - 迭代配置：`Flux2_lora_v7.json`, `Flux2_lora_v8.json`, `Flux2_lora_v9.json`, ...
  - 每一版配置与上一版配置对比，同时与 v6 起点配置对比。
- **评估指标**:
  - 主要指标：测试集 LPIPS mean，数值越低越好。
  - 次要记录：训练 loss 曲线、训练步数、训练是否完整结束、生成样例路径、评估脚本 stdout/stderr、报错信息。
- **实验设置**:
  - 训练工作目录：`/root/autodl-tmp/ai-toolkit`
  - 训练环境：在该目录下执行 `source venv/bin/activate`
  - 初始训练命令：
    ```bash
    cd /root/autodl-tmp/ai-toolkit
    source venv/bin/activate
    python run.py config/Flux2_lora_v6.json
    ```
  - LoRA 权重复制规则：
    1. 每轮训练结束后，定位该轮输出的 `Flux2_lora_v*.safetensors`。
    2. 将该权重复制到：
       ```bash
       /root/autodl-tmp/ComfyUI/models/loras/new_lora.safetensors
       ```
    3. 复制后应确认文件存在且大小大于 0。
  - 评估工作目录：`/root/autodl-tmp/ComfyUI`
  - 评估环境：使用名为 `base` 的 conda 环境。
  - 评估命令：
    ```bash
    cd /root/autodl-tmp/ComfyUI
    conda activate base
    python comfy_api_lora_lpips_eval.py
    ```
  - 每轮必须从评估脚本输出或结果文件中提取整个测试集 LPIPS 均值。
- **成功标准**:
  - 流程成功完成至少 3 个版本：v6、v7、v8。
  - 每轮均能得到明确的 LPIPS mean。
  - 至少一个后续版本的 LPIPS mean 低于 v6。
  - 优先保留 LPIPS mean 最低的版本及其配置。
- **失败解读**:
  - 如果训练失败，优先检查环境激活、JSON 路径、模型路径、数据路径、显存不足。
  - 如果找不到输出 LoRA，说明输出路径或命名规则需要修正，停止迭代并记录证据。
  - 如果评估脚本失败，优先检查 ComfyUI 是否可调用、`new_lora.safetensors` 是否复制成功、测试集路径是否正确。
  - 如果后续版本 LPIPS 未下降，说明当前修改方向无效，应回滚到历史最优配置并小步调整其他参数。
- **优先级**: 必须运行

### 实验块 2: 参数优化实验——基于 LPIPS 的 JSON 配置迭代
- **验证 Claim**: C1, C3
- **对比系统**:
  - `Flux2_lora_v6.json` vs. `Flux2_lora_v7.json`
  - 历史最优版本 vs. 新候选版本
- **允许修改的内容**:
  - 仅修改 `/root/autodl-tmp/ai-toolkit/config/Flux2_lora_v*.json` 中与训练超参数、LoRA 网络参数、优化器、学习率、训练步数、batch size、采样/保存频率相关的字段。
  - 每次生成新版本配置文件时，应从历史最优版复制得到，严禁覆盖旧版本。
  - 新版本必须同步更新配置内部的输出名称，使输出权重能区分版本，例如 `Flux2_lora_v7`、`Flux2_lora_v8`。
- **禁止修改的内容**:
  - 不要修改 AI-Toolkit 源码。
  - 不要修改 `comfy_api_lora_lpips_eval.py` 的评估逻辑。
  - 不要修改测试集。
  - 不要把不同版本的评估结果混在一起。
  - 不要覆盖历史 JSON 配置。
- **参数搜索策略**:
  - 采用小步、可解释的单轮修改，优先一次只改变少量关键参数。
  - 每次修改后必须记录：
    - 修改字段
    - 修改前数值
    - 修改后数值
    - 修改理由
    - 期望改善方向
  - 如果某次修改导致 LPIPS 变差，下一轮应回滚该修改或基于历史最优版本继续。
- **成功标准**:
  - 每个新版本配置文件均能追溯到上一版本。
  - 每轮修改都有明确理由。
  - 至少找到一个 LPIPS mean 低于 v6 的配置。
- **优先级**: 必须运行

### 实验块 3: 结果记录与审计
- **验证 Claim**: C2, C3
- **记录文件建议**:
  - `refine-logs/LOOP_RESULTS.md`
  - `refine-logs/LPIPS_HISTORY.csv`
  - `refine-logs/CONFIG_DIFFS.md`
- **每轮记录格式**:

| Version | Config Path | Train Command | LoRA Source Path | Copied LoRA Path | LPIPS Mean | Compared To Previous | Compared To v6 | Decision | Notes |
|---------|-------------|---------------|------------------|------------------|------------|----------------------|----------------|----------|-------|
| v6 | `/root/autodl-tmp/ai-toolkit/config/Flux2_lora_v6.json` | `python run.py config/Flux2_lora_v6.json` | 待填写 | `/root/autodl-tmp/ComfyUI/models/loras/new_lora.safetensors` | 待填写 | - | - | baseline | 初始配置 |
| v7 | `/root/autodl-tmp/ai-toolkit/config/Flux2_lora_v7.json` | `python run.py config/Flux2_lora_v7.json` | 待填写 | `/root/autodl-tmp/ComfyUI/models/loras/new_lora.safetensors` | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |

- **成功标准**:
  - 每一轮都有完整记录。
  - 最终能明确指出历史最优版本、最低 LPIPS mean、对应配置文件、对应 LoRA 权重。
- **优先级**: 必须运行

## 运行顺序

| 里程碑 | 目标 | 运行内容 | 决策关卡 | 预估耗时 |
|--------|------|---------|---------|---------|
| M0: 环境健全性检查 | 确认训练和评估环境都能启动 | 检查 `/root/autodl-tmp/ai-toolkit`、`source venv/bin/activate`、`Flux2_lora_v6.json`、`/root/autodl-tmp/ComfyUI`、`conda activate base`、`comfy_api_lora_lpips_eval.py` | 两个环境均可进入？配置文件和脚本均存在？ | ~0.2h |
| M1: v6 基线训练 | 得到初始 LoRA 权重 | 执行 `python run.py config/Flux2_lora_v6.json` | 训练是否正常结束？是否生成 `Flux2_lora_v6.safetensors` 或对应 safetensors 文件？ | 取决于配置 |
| M2: v6 基线评估 | 得到初始 LPIPS mean | 将 v6 权重复制为 `new_lora.safetensors`，运行 `python comfy_api_lora_lpips_eval.py` | 是否得到整个测试集 LPIPS mean？ | 取决于测试集规模 |
| M3: 生成 v7 配置 | 基于 v6 指标进行第一次参数优化 | 复制 `Flux2_lora_v6.json` 为 `Flux2_lora_v7.json`，小步修改参数，并记录 diff | 修改是否合理且可解释？ | ~0.2h |
| M4: v7 训练与评估 | 验证第一次优化是否有效 | 训练 v7，复制 v7 LoRA，运行 LPIPS 评估 | v7 LPIPS 是否低于 v6？ | 取决于配置 |
| M5: 继续迭代 | 生成 v8、v9 等版本 | 基于历史最优版本继续小步优化 | 是否继续下降？连续 3 轮不下降则停止 | 取决于迭代次数 |
| M6: 总结最优结果 | 输出最终可用配置 | 汇总所有版本 LPIPS、配置差异和最优 LoRA 路径 | 是否能明确推荐一个最优版本？ | ~0.2h |

## 算力预算

- **预估总 GPU 小时**: 每轮约7h（例如，迭代完v6为一轮）。
- **硬件**: 1x RTX PRO 6000(96GB)
- **最大瓶颈**:
  - Flux2 LoRA 单轮训练耗时。
  - ComfyUI 评估脚本生成完整测试集结果的耗时。
  - 如果每轮修改幅度过大，会浪费 GPU 时间。
- **建议迭代预算**:
  - 第一阶段：v6、v7、v8，至少 3 轮，验证闭环有效。
  - 第二阶段：在历史最优版本附近继续 v9、v10，直到 LPIPS 连续 3 轮无改进，或达到用户指定的最大训练成本。

## 风险

- **风险**: 训练环境未激活，导致依赖缺失 → **缓解措施**: 在 `/root/autodl-tmp/ai-toolkit` 中先执行 `source venv/bin/activate`，再运行训练命令。
- **风险**: 评估环境与训练环境不同，导致包冲突 → **缓解措施**: 训练只用 AI-Toolkit 的 `venv`，评估只用 conda 的 `base` 环境，避免混用。
- **风险**: 找不到训练输出的 safetensors 文件 → **缓解措施**: 在 `/root/autodl-tmp/ai-toolkit/output/` 及相关输出目录中搜索最新的 `Flux2_lora_v*.safetensors`；若存在多个候选文件，按版本名和修改时间确认；仍无法确认则停止，不要随意复制错误权重。
- **风险**: LPIPS 评估输出格式不固定 → **缓解措施**: 优先从评估脚本 stdout、日志文件、CSV 或 JSON 中提取 `LPIPS mean`；无法提取时保存完整输出并停止迭代。
- **风险**: 过度修改 JSON，导致实验不可解释 → **缓解措施**: 每轮只允许小步修改少量关键训练参数，并在 `CONFIG_DIFFS.md` 中记录修改理由。
- **风险**: 连续训练造成磁盘占用过高 → **缓解措施**: 每轮记录权重路径和日志路径，保留历史最优与最近版本，清理明显失败的中间缓存前必须先记录。

