# M2M-LINet

面向遥感影像二值变化检测的轻量级交错网络。本项目基于 [M2M-LINet](https://github.com/Sean1005-x/M2M-LINet) 整理，提供统一的数据读取、训练、评估和预测流程，支持 CPU 与单卡 CUDA。

论文：[From Macro to Micro: A Lightweight Interleaved Network for Remote Sensing Image Change Detection](https://doi.org/10.1109/TGRS.2025.3548562)，IEEE TGRS，2025。

## 项目结构

```text
M2M-LINet/
├── m2m_linet/
│   ├── models/          # PVTv2-B0、LCB、LTB、BAEM 与主网络
│   ├── data.py          # 影像配对、二值标签、同步增强、边缘标签
│   ├── losses.py        # 变化 BCE + Dice，边缘 Dice
│   ├── metrics.py       # 全数据集混淆矩阵与二值指标
│   ├── checkpoint.py    # 权重加载、断点保存、随机状态恢复
│   ├── engine.py        # 共用训练、评估、预测逻辑
│   └── cli.py           # 参数与运行配置
├── pretrained/         # 本地骨干预训练权重（不纳入 Git）
├── tests/              # 数据、指标、损失、模型及工作流测试
├── train.py
├── evaluate.py
└── predict.py
```

## 安装

Python 3.9 或以上。建议创建独立虚拟环境，在项目根目录执行：

```bash
python -m pip install -r requirements.txt
```

CUDA 环境请先按 [PyTorch 安装说明](https://pytorch.org/get-started/locally/) 安装与机器匹配的 PyTorch，再安装其余依赖。此项目无需 `timm`、`thop`、`torchvision` 或 TensorBoard。

将 PVTv2-B0 预训练权重放在 `pretrained/pvt_v2_b0.pth`，或通过 `--backbone-weights` 指定位置；下载位置和校验值见 [预训练权重说明](pretrained/README.md)。全新训练时如需随机初始化，显式传入 `--from-scratch`。

## 数据格式

```text
DATA_ROOT/
├── train/
│   ├── A/       # 时相一 RGB 影像
│   ├── B/       # 时相二 RGB 影像
│   ├── label/   # 二值变化标签
│   └── edge/    # 可选，二值边缘标签
├── val/
│   └── A/ B/ label/ [edge/]
└── test/
    └── A/ B/ label/ [edge/]
```

也支持用 `time1/time2` 替代 `A/B`；同一划分中不要同时放两套时相目录。影像和标签按文件主名配对，例如 `A/0001.jpg`、`B/0001.png`、`label/0001.png`。读取时检查缺失配对、重复主名、影像尺寸和标签取值，避免静默错配。

- 影像转换为 RGB，归一化到 `[-1, 1]`。
- 标签支持 `{0,1}` 或 `{0,255}`，白色/1 表示变化；不自动映射其他类别。
- 输入高、宽均需至少 64，且为 32 的倍数；推荐预先裁为 256 × 256。程序不自动缩放影像或标签。
- 未提供 `edge/` 时，从变化标签生成 3 × 3 形态学梯度边缘，边界采用复制填充；如已有边缘真值，可直接使用。
- `--augment` 启用成对同步增强；默认关闭。每个 batch 内影像尺寸必须一致。

## 训练与恢复

训练使用 `train`，每轮在独立的 `val` 上计算指标，以变化类 F1 保存最优模型。`test` 仅用于训练结束后的评估。

```bash
python train.py --data-root /path/to/DATA_ROOT --epochs 200 --batch-size 16 --device cuda:0 --output-dir runs/experiment
```

默认配置：AdamW、学习率 `1e-4`、weight decay `5e-4`、betas `(0.9, 0.999)`、常量学习率、seed `2022`；主任务损失为 BCEWithLogits + Dice，边缘辅助损失为 Dice（权重 `1.0`）。这些配置均写入运行目录，batch size、归一化、边缘生成与增强等为本项目的运行配置，不代表已获得论文结果。需要修改时使用 `--lr`、`--weight-decay`、`--edge-weight` 等参数；完整列表见 `python train.py --help`。

```bash
python train.py --data-root /path/to/DATA_ROOT --epochs 200 --batch-size 16 --device cuda:0 --output-dir runs/experiment --resume runs/experiment/last.pt
```

恢复时沿用原来的优化器、batch size、随机种子、增强、AMP、worker 数等参数。可增加总 epoch 数；保存于 epoch 结束，不支持从一个 epoch 中间恢复。若要更严格的可重复性，可在初次训练和恢复时都传入 `--deterministic`。`--amp` 为 CUDA 混合精度可选项，默认关闭。

运行目录包括：

- `best.pt`：验证集 F1 最高的完整检查点；F1 相同时保留最早的一次。
- `last.pt`：最近完成 epoch 的模型、优化器、AMP、随机状态与配置。
- `config.json`：本次运行的参数。
- `history.jsonl`、`train.log`：逐轮指标及日志。

程序支持 CPU 和单卡 CUDA，不再隐式要求两张 GPU。Windows 如遇 worker 环境问题，可使用 `--num-workers 0`。换机器或 PyTorch/CUDA 版本不保证逐位相同。

## 评估与预测

```bash
python evaluate.py --data-root /path/to/DATA_ROOT --split test --checkpoint runs/experiment/best.pt --device cuda:0 --metrics-output runs/experiment/test_metrics.json
python predict.py --data-root /path/to/DATA_ROOT --split test --checkpoint runs/experiment/best.pt --device cuda:0 --output-dir predictions/experiment
```

评估输出变化类 Precision、Recall、F1、IoU，以及全像素 Accuracy、Cohen's Kappa；阈值为 `0.5`，使用全划分累计混淆矩阵。分母为零时对应指标定义为 `0`。

预测只需要两个时相目录，不要求标签。输出按输入文件主名保存为 0/255 PNG；可用 `--threshold` 修改阈值，`--save-probabilities` 额外保存 float32 概率数组。输出文件同名时会被覆盖，请为不同实验选择不同目录。

## 模型接口

```python
import torch
from m2m_linet.models import M2MLINet

model = M2MLINet()
model.eval()
with torch.inference_mode():
    change_logits, edge_logits = model(torch.randn(1, 3, 256, 256), torch.randn(1, 3, 256, 256))
    change_probability = change_logits.sigmoid()
```

两个输出均为 `[N, 1, H, W]` logits；损失直接接收 logits，评估与预测内部只执行一次 sigmoid。`evaluate.py` 和 `predict.py` 可读取标准化检查点或官方主模型权重，自动处理 `module.` 前缀和已移除的无效参数；结构不兼容的实验变体权重会明确报错。

## 开发检查

```bash
python -m unittest discover -s tests -v
python -m compileall -q m2m_linet train.py evaluate.py predict.py
python -m pip install -r requirements-dev.txt
python -m black --check .
python -m pycodestyle m2m_linet tests train.py evaluate.py predict.py
```

CI 会执行格式检查与单元测试。完整训练精度需要在实际数据集上重新测量，单元测试和运行冒烟测试不代替精度复现。

## 来源与引用

模型来源为 Sean1005-x/M2M-LINet，基准提交为 [`bd57500`](https://github.com/Sean1005-x/M2M-LINet/tree/bd57500cefa3e4335c7ddceab0d5d48e90bb9cfd)。保留原作者署名及论文引用；上游未提供独立 LICENSE 文件，本整理版本不额外声明开源许可。

```bibtex
@ARTICLE{10912653,
  author={Xu, Yetong and Lei, Tao and Ning, Hailong and Lin, Shaoxiong and Liu, Tongfei and Gong, Maoguo and Nandi, Asoke K.},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  title={From Macro to Micro: A Lightweight Interleaved Network for Remote Sensing Image Change Detection},
  year={2025},
  volume={63},
  pages={1-14},
  doi={10.1109/TGRS.2025.3548562}
}
```
