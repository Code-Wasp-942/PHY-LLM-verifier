# 项目操作指南（中文）

本文档用于指导你在当前项目中完成数据准备、SFT 训练、生成预测和评估。

## 1. 项目结构

- `configs/`：模型与训练配置
- `src/data/`：数据转换脚本
- `src/train/`：SFT 训练入口
- `src/eval/`：预测与评估脚本
- `scripts/`：一键命令脚本
- `data/raw/`：原始数据
- `data/processed/`：处理后的 `train/val/test.jsonl`
- `checkpoints/`：训练输出模型
- `outputs/`：预测结果
- `reports/`：评估报告（可选）
- `logs/`：日志（可选）

## 2. 环境准备

建议使用 Python 3.10+。

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -e .
```

如果你需要开发依赖（pytest/ruff）：

```bash
pip install -e .[dev]
```

## 3. 准备数据

将原始数据放到 `data/raw/`，例如：`data/raw/verifybench.json`。

执行：

```bash
bash scripts/01_prepare_data.sh data/raw/verifybench.json
```

默认会输出：

- `data/processed/train.jsonl`
- `data/processed/val.jsonl`
- `data/processed/test.jsonl`

## 4. 启动训练（Qwen2.5-8B-Instruct + DeepSpeed）

默认训练命令：

```bash
bash scripts/02_train_sft.sh
```

默认读取配置：

- 训练配置：`configs/train/sft.yaml`
- 模型配置：`configs/model/qwen2_5_8b_sft.yaml`
- DeepSpeed 配置：`configs/train/deepspeed_zero2.json`

断点续训：

```bash
bash scripts/02_train_sft.sh checkpoints/sft_full/checkpoint-xxxx
```

## 5. 生成预测

使用训练后的模型对测试集生成预测：

```bash
bash scripts/03_generate_pred.sh checkpoints/sft_full data/processed/test.jsonl outputs/predictions.jsonl
```

## 6. 评估结果

执行评估：

```bash
bash scripts/04_eval.sh outputs/predictions.jsonl data/processed/test.jsonl
```

当前评估指标：

- `point_precision`
- `point_recall`
- `point_f1`
- `exact_set_match`
- `score_mae`

## 7. 常见问题

1) 找不到数据文件

- 请确认 `data/processed/train.jsonl` 等文件已生成。

2) 显存不足

- 可在 `configs/train/sft.yaml` 中降低 `per_device_train_batch_size`。
- 增大 `gradient_accumulation_steps` 保持等效 batch。

3) 多卡命令不匹配本机卡数

- 修改 `scripts/02_train_sft.sh` 中 `--nproc_per_node=6` 为实际 GPU 数量。

4) 无法启动 DeepSpeed

- 请确认 `deepspeed` 与 `torch` 版本兼容，必要时重装依赖。

## 8. 推荐执行顺序

```bash
bash scripts/01_prepare_data.sh data/raw/verifybench.json
bash scripts/02_train_sft.sh
bash scripts/03_generate_pred.sh checkpoints/sft_full data/processed/test.jsonl outputs/predictions.jsonl
bash scripts/04_eval.sh outputs/predictions.jsonl data/processed/test.jsonl
```
