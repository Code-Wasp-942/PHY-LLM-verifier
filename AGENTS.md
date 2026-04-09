# AGENTS.md

本文档定义本仓库内编码代理的统一工作规范。
作用范围：当前目录及其全部子目录。

## 项目概览

- 目标：面向物理问答验证任务的 SFT + 评测流水线。
- 主要语言：Python（`src/`）。
- 配置位置：`configs/` 下的 YAML 文件。

## 关键入口

- 训练：`src/train/run_sft.py`
- 数据准备：`src/data/build_sft_dataset.py`
- 预测：`src/eval/generate_predictions.py`
- 指标计算：`src/eval/metrics.py`

## 常用命令

- 准备数据：`bash scripts/01_prepare_data.sh`
- 训练 SFT：`bash scripts/02_train_sft.sh`
- 执行评测：`bash scripts/04_eval.sh`
- 运行测试：`pytest -q`
- 代码检查（按需）：`ruff check .`

## 配置规范

- 模型路径配置应集中在 `configs/model/` 下的模型配置文件中。
- 优先使用 `configs/train/sft.yaml` 通过 `model_config` 引用模型配置。
- 除非确有必要，避免在多个配置文件中重复定义 `model_name_or_path`。

## 编码约定

- 遵循现有代码风格，改动保持最小且聚焦。
- 优先使用语义清晰的变量名，避免单字母命名。
- 适当添加注释，以代码块为单位。
- 默认使用 ASCII，除非文件已明确使用非 ASCII。

## 安全与边界

- 每次编辑后使用 `git diff`，并报告更改内容。
- 未经明确要求，不要修改以下生成产物目录：
  - `checkpoints/`
  - `outputs/`
  - `logs/`
  - `reports/`
- 不要提交密钥、凭据或本地环境文件。
- 不要执行破坏性 git 命令（例如 `git reset --hard`）。

## 验证建议

- 对代码改动，先执行最小且相关的验证。
- 若改动训练配置或加载逻辑，优先通过训练入口命令验证。
- 若改动评测代码，使用样例文件运行 `scripts/04_eval.sh` 验证。
