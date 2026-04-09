from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments


@dataclass
class SFTConfig:
    model_name_or_path: str
    train_file: str
    val_file: str
    output_dir: str
    max_length: int = 2048
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    num_train_epochs: float = 1.0
    max_steps: int = -1
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    bf16: bool = True
    gradient_checkpointing: bool = True
    deepspeed: str | None = None
    report_to: str = "none"
    seed: int = 42
    trust_remote_code: bool = True
    torch_dtype: str | None = None
    optim: str = "adamw_torch"
    max_grad_norm: float = 1.0
    dataloader_num_workers: int = 2
    save_safetensors: bool = True


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data or {}


def _load_config(train_config_path: Path, model_config_path: Path | None) -> SFTConfig:
    train_data = _load_yaml(train_config_path)
    model_config_from_train = train_data.pop("model_config", None)

    if model_config_path is None and model_config_from_train:
        candidate = Path(model_config_from_train)
        if not candidate.is_absolute():
            candidate = (train_config_path.parent / candidate).resolve()
        model_config_path = candidate

    model_data = _load_yaml(model_config_path) if model_config_path else {}
    merged = {**model_data, **train_data}
    return SFTConfig(**merged)


def _resolve_torch_dtype(config: SFTConfig) -> torch.dtype:
    if config.torch_dtype is None:
        return torch.bfloat16 if config.bf16 else torch.float32

    dtype_map = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    normalized = str(config.torch_dtype).strip().lower()
    if normalized not in dtype_map:
        raise ValueError(f"Unsupported torch_dtype: {config.torch_dtype}")
    return dtype_map[normalized]


def _load_jsonl(path: str) -> Dataset:
    if not Path(path).exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    rows: List[Dict] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return Dataset.from_list(rows)


def _truncate_pair(
    prompt_ids: List[int], target_ids: List[int], max_length: int
) -> tuple[List[int], List[int]]:
    total = len(prompt_ids) + len(target_ids) + 1
    if total <= max_length:
        return prompt_ids, target_ids

    overflow = total - max_length

    if overflow < len(prompt_ids):
        prompt_ids = prompt_ids[overflow:]
        return prompt_ids, target_ids

    overflow -= len(prompt_ids)
    prompt_ids = []
    if overflow < len(target_ids):
        target_ids = target_ids[overflow:]
    else:
        target_ids = target_ids[-1:]
    return prompt_ids, target_ids


def _build_tokenizer_and_model(config: SFTConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = _resolve_torch_dtype(config)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
        torch_dtype=dtype,
    )
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    return tokenizer, model


def _tokenize_sample(example: Dict, tokenizer, max_length: int) -> Dict:
    prompt = example["prompt"]
    target = example["target"]

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    prompt_ids, target_ids = _truncate_pair(prompt_ids, target_ids, max_length)

    eos_id = tokenizer.eos_token_id
    input_ids = prompt_ids + target_ids + [eos_id]
    labels = ([-100] * len(prompt_ids)) + target_ids + [eos_id]
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class DataCollatorForCausalLM:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in features)

        input_ids = []
        attention_mask = []
        labels = []

        for item in features:
            pad_len = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(item["attention_mask"] + [0] * pad_len)
            labels.append(item["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full-parameter SFT for Qwen 8B")
    parser.add_argument("--config", type=Path, default=Path("configs/train/sft.yaml"))
    parser.add_argument("--model-config", type=Path, default=None)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    args = parser.parse_args()

    model_config_path = (
        args.model_config if args.model_config and args.model_config.exists() else None
    )
    config = _load_config(args.config, model_config_path)
    tokenizer, model = _build_tokenizer_and_model(config)

    train_dataset = _load_jsonl(config.train_file)
    val_dataset = _load_jsonl(config.val_file) if Path(config.val_file).exists() else None

    train_dataset = train_dataset.map(
        lambda row: _tokenize_sample(row, tokenizer, config.max_length),
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train dataset",
    )
    if val_dataset is not None:
        val_dataset = val_dataset.map(
            lambda row: _tokenize_sample(row, tokenizer, config.max_length),
            remove_columns=val_dataset.column_names,
            desc="Tokenizing validation dataset",
        )

    training_kwargs: Dict[str, Any] = {
        "output_dir": config.output_dir,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.num_train_epochs,
        "max_steps": config.max_steps,
        "warmup_ratio": config.warmup_ratio,
        "weight_decay": config.weight_decay,
        "lr_scheduler_type": config.lr_scheduler_type,
        "optim": config.optim,
        "max_grad_norm": config.max_grad_norm,
        "dataloader_num_workers": config.dataloader_num_workers,
        "logging_steps": config.logging_steps,
        "save_steps": config.save_steps,
        "eval_steps": config.eval_steps,
        "save_total_limit": config.save_total_limit,
        "bf16": config.bf16,
        "gradient_checkpointing": config.gradient_checkpointing,
        "deepspeed": config.deepspeed,
        "report_to": [] if config.report_to == "none" else [config.report_to],
        "seed": config.seed,
        "save_strategy": "steps",
        "logging_strategy": "steps",
        "remove_unused_columns": False,
        "load_best_model_at_end": False,
        "ddp_find_unused_parameters": False,
    }

    training_arg_fields = inspect.signature(TrainingArguments.__init__).parameters
    eval_strategy_value = "steps" if val_dataset is not None else "no"
    if "eval_strategy" in training_arg_fields:
        training_kwargs["eval_strategy"] = eval_strategy_value
    elif "evaluation_strategy" in training_arg_fields:
        training_kwargs["evaluation_strategy"] = eval_strategy_value

    if "save_safetensors" in training_arg_fields:
        training_kwargs["save_safetensors"] = config.save_safetensors

    training_args = TrainingArguments(**training_kwargs)

    collator = DataCollatorForCausalLM(pad_token_id=tokenizer.pad_token_id)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    print(
        json.dumps(
            {
                "train_file": config.train_file,
                "val_file": config.val_file,
                "model_name_or_path": config.model_name_or_path,
                "output_dir": config.output_dir,
                "max_length": config.max_length,
                "resume_from_checkpoint": args.resume_from_checkpoint,
            },
            ensure_ascii=False,
        )
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)


if __name__ == "__main__":
    main()
