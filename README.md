# AEQ-DETR: Adaptive Evidence Querying via Sampling and Reallocation for Multispectral Object Detection

[中文](README_CN.md)

**The full training and inference code, along with the model checkpoints used in the paper, will be released upon acceptance of the paper.**

## Experimental environment

| Component | Environment |
| --- | --- |
| OS | Linux |
| Python | 3.11.10 |
| GPU | NVIDIA GeForce RTX 3090, 24 GB |
| PyTorch | 2.4.0+cu121 |
| PyTorch CUDA | 12.1 |

## Training logs

The prefixes `aeq_` and `aeq_p_` denote AEQ-DETR and AEQ-DETR-P, respectively. All logs use seed 0 and contain 20 epochs for FLIR and LLVIP, and 50 epochs for M3FD.

Each CSV row contains one epoch's `epoch`, `learning_rate`, training losses, and COCO AP/AR metrics. Loss columns vary by file. AP/AR values use a 0–1 scale; multiply by 100 for percentage scores. A value of `-1` indicates an unavailable metric.

Run this example from the repository root to read the FLIR AEQ-DETR metrics at the epoch with the highest AP:

```python
import csv

with open("train_logs/flir/aeq_flir_seed0_20e.csv", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
best = max(rows, key=lambda row: float(row["AP"]))
print("epoch:", best["epoch"])
for metric in ("AP", "AP50", "AP75"):
    print(f"{metric}: {float(best[metric]) * 100:.2f}")
```

For curves, plot losses or AP/AR against `epoch`. Read metrics from the same row when comparing results at a given epoch.

## License

See [LICENSE](LICENSE) and [NOTICE](NOTICE) for licensing and upstream attribution.
