# PVTv2-B0 预训练权重

本地交付目录已放入原工程的 `pvt_v2_b0.pth`，其 Git blob 与上游一致。Git 中忽略权重文件，以免重复提交二进制运行资产。

从仓库拉取代码后，在项目根目录执行：

```bash
python -c "from urllib.request import urlretrieve; urlretrieve('https://raw.githubusercontent.com/Sean1005-x/M2M-LINet/bd57500cefa3e4335c7ddceab0d5d48e90bb9cfd/pvt_v2_b0.pth', 'pretrained/pvt_v2_b0.pth')"
python -c "import hashlib; from pathlib import Path; p=Path('pretrained/pvt_v2_b0.pth'); expected='fbb931ad59ab4d64e3f4370991326803e3af882fcaa877b81550d368d1fbab1f'; assert hashlib.sha256(p.read_bytes()).hexdigest()==expected; print('SHA256 OK')"
```

权重只在显式训练初始化时读取；评估、预测和断点恢复直接使用完整模型检查点，不重复加载骨干权重。
