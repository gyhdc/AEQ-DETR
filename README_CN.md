# AEQ-DETR: Adaptive Evidence Querying via Sampling and Reallocation for Multispectral Object Detection

[English](README.md)

**完整训练与推理代码，以及论文实验对应的模型权重，将在论文接收后公开。**

## 实验环境

| 项目 | 环境 |
| --- | --- |
| 操作系统 | Linux |
| Python | 3.11.10 |
| GPU | NVIDIA GeForce RTX 3090，24 GB |
| PyTorch | 2.4.0+cu121 |
| PyTorch CUDA | 12.1 |

## 训练日志

文件名前缀 `aeq_` 对应 AEQ-DETR，`aeq_p_` 对应 AEQ-DETR-P，均为 seed 0。FLIR 和 LLVIP 各包含 20 轮记录，M3FD 包含 50 轮记录。

CSV 每行对应一轮，记录 `epoch`、`learning_rate`、训练损失及 COCO AP/AR 指标；具体损失字段以各文件表头为准。AP/AR 使用 0–1 标度，乘以 100 可转换为百分制；`-1` 表示无有效评估值。

在仓库根目录运行以下代码，可查看 FLIR 的 AEQ-DETR 日志中 AP 最高轮次的指标：

```python
import csv

with open("train_logs/flir/aeq_flir_seed0_20e.csv", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
best = max(rows, key=lambda row: float(row["AP"]))
print("epoch:", best["epoch"])
for metric in ("AP", "AP50", "AP75"):
    print(f"{metric}: {float(best[metric]) * 100:.2f}")
```

绘制曲线时，以 `epoch` 为横轴，损失或 AP/AR 为纵轴。比较同一轮的指标时，读取同一行数据。

## 许可

许可及上游来源见 [LICENSE](LICENSE) 和 [NOTICE](NOTICE)。
