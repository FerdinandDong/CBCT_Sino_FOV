## 🧱 当前代码结构（2025‑08‑28）

> 下列模块今天已建立或已跑通（含 Dummy 空跑）：

```
CBCT_Sino_FOV/
├─ configs/
│  ├─ train_unet.yaml        # 训练配置（支持 Dummy / 真实数据）
│  └─ eval.yaml              # 评估配置（PSNR/SSIM）
├─ scripts/
│  ├─ train.py               # 统一训练入口（UTF-8 读取 cfg；模型注册触发）
│  └─ evaluate.py            # 统一评估入口（支持加载权重，Dummy 回退）
├─ ctprojfix/
│  ├─ data/
│  │  └─ dataset.py          # ProjectionAnglesDataset（遍历 360 角度）；DummyDataset
│  ├─ models/
│  │  ├─ registry.py         # 模型注册表（build_model/装饰器 register）
│  │  └─ unet.py             # UNet baseline（in_ch=2/3，out_ch=1）
│  ├─ trainers/
│  │  └─ supervised.py       # L1 损失，Adam；类型强制转换；可保存 ckpt（预留）
│  └─ evals/
│     └─ metrics.py          # PSNR & SSIM（简化版）
├─ pyproject.toml            # setuptools 可编辑安装（仅打包 ctprojfix）
└─ README.md                 # 本文件（实验思路 + 日志）
```

---

## 目标结构（规划 & 可插拔）
```
```


## 使用方法（Quick Start）

### 安装（可编辑）
```bash
pip install -e .
```

### 训练（本地无数据 → Dummy 空跑）
```bash
python scripts/train.py --cfg configs/train_unet.yaml
```
- `configs/train_unet.yaml` 中：
  - `data.use_dummy: true`（本地无数据时）
  - 若开启 `data.add_angle_channel: true`，请将 `model.params.in_ch` 设为 3

### 评估（PSNR/SSIM）
```bash
python scripts/evaluate.py --cfg configs/eval.yaml
```
- `eval.ckpt` 可填训练权重路径（为空则用随机权重，仅作流程检查）

---

## Changelog

### 2025‑08‑28
- 初始化仓库 & 目录骨架（configs/scripts/ctprojfix/...）
- `pyproject.toml` 完成（仅打包 ctprojfix；Python ≥3.9）
- 建立 **模型注册表 registry**；实现 **UNet baseline**
- 训练入口 `scripts/train.py`（UTF-8 读 cfg；显式触发注册）
- 实现 **SupervisedTrainer**（L1 + Adam；类型安全）
- 实现 **DummyDataset**（本地无数据可完整跑通训练）
- 新增 **PSNR/SSIM**（简化版）与 `scripts/evaluate.py`
- **ProjectionAnglesDataset**：按服务器尺寸 `(360,960,1240)` 逐角度遍历；支持 step/downsample/mask 模式
- 本地验证：Dummy 训练 2 个 epoch，loss 正常下降，评估脚本可运行

> 明日计划：加入 checkpoint 保存/加载；补充重建脚本（FBP/FDK）与指标闭环。

---

## 备注（编码/跨平台）
- Windows 下读取 YAML 建议：`open(..., encoding="utf-8")`（已在脚本中处理）服务器Linux似乎需要更改,检查一下
- 大文件/权重/图片/视频已在 `.gitignore` 中忽略，避免误传仓库
- Torch/CUDA 建议按各自平台单独安装（未写入 `pyproject.toml` 依赖）

---