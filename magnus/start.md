# Docker 内仅 `pip` 权限部署流程

本文档适用于：你在 Docker 容器内没有 `apt` 权限，只能安装 `pip` 包。

## 1) 推荐基础镜像

优先使用（已带 CUDA + Python + PyTorch 的镜像）：

```bash
docker pull pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel
```

## 2) 启动容器并挂载项目

示例：

```bash
docker run --gpus all -it --rm \
  -v /path/to/demo-01-verifier:/workspace/demo-01-verifier \
  -w /workspace/demo-01-verifier \
  pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel bash
```

## 3) 仅用 pip 安装依赖

先升级基础安装工具：

```bash
python -m pip install -U pip setuptools wheel
```

安装项目核心依赖（兼容本仓库当前版本）：

```bash
pip install \
  "transformers==4.52.4" \
  "datasets==3.6.0" \
  "huggingface-hub==0.36.2" \
  "fsspec[http]==2025.3.0" \
  "numpy>=1.26.0" \
  "pydantic>=2.7.0" \
  "pyyaml>=6.0.1" \
  "scikit-learn>=1.4.0"
```

安装 DeepSpeed（仅 pip 权限时优先关闭自定义算子编译）：

```bash
DS_BUILD_OPS=0 pip install "deepspeed==0.14.4"
```

安装当前项目：

```bash
pip install -e .
```

可选开发工具：

```bash
pip install pytest ruff
```

## 4) 验证环境

```bash
python -c "import torch, transformers, deepspeed; print(torch.__version__)"
python -m src.train.run_sft --help
python -m src.data.build_sft_dataset --help
```

## 5) 模型权重准备（本地目录）

如果你已下载模型到本地目录，例如：

- `./models/Qwen2.5-8B-Instruct`

当前配置已指向本地路径：

- `configs/train/sft.yaml`
- `configs/model/qwen2_5_8b_sft.yaml`

## 6) 运行流程

```bash
bash scripts/01_prepare_data.sh data/raw/verifybench.json
bash scripts/02_train_sft.sh
bash scripts/03_generate_pred.sh checkpoints/sft_full data/processed/test.jsonl outputs/predictions.jsonl
bash scripts/04_eval.sh outputs/predictions.jsonl data/processed/test.jsonl
```

## 7) 常见问题

1. `deepspeed` 安装失败

- 先确认是 `DS_BUILD_OPS=0` 方式安装。
- 若仍失败，通常是镜像与 CUDA/PyTorch 组合不兼容，建议更换到同系列 `pytorch/pytorch` 镜像。

2. `transformers` 导入报依赖版本错误

- 执行：

```bash
pip install "huggingface-hub==0.36.2" "fsspec[http]==2025.3.0"
```

3. 训练命令找不到数据

- 确认已先执行 `scripts/01_prepare_data.sh`，并生成 `data/processed/train.jsonl`。

## 8) 备选镜像方案 B（`huggingface/transformers-all-latest-gpu`）

如果你希望直接使用该镜像，建议不要用其默认浮动依赖，进入容器后立刻锁版本：

```bash
python -m pip install -U pip setuptools wheel

# 可选：先清理潜在冲突
pip uninstall -y transformers huggingface-hub datasets fsspec deepspeed || true

# 安装与本项目兼容的锁版本
pip install \
  "transformers==4.52.4" \
  "huggingface-hub==0.36.2" \
  "datasets==3.6.0" \
  "fsspec[http]==2025.3.0" \
  "numpy>=1.26.0" \
  "pydantic>=2.7.0" \
  "pyyaml>=6.0.1" \
  "scikit-learn>=1.4.0"

# 仅 pip 权限下安装 deepspeed
DS_BUILD_OPS=0 pip install "deepspeed==0.14.4"

# 安装项目
pip install -e .

# 校验
python -m pip check
python -c "import torch, transformers, datasets, deepspeed; print('ok')"
python -m src.train.run_sft --help
```
