"""
Train the value model (Llama-3.2-1B) to predict discounted distance to QED.

Labels are gamma^d where d = remaining proof steps, gamma = 0.95.
States near completion have labels near 1.0, failed states have label 0.0.

The model is trained as a text-to-score model via SFT:
  Input: "Estimate the discounted distance...\n{state}\nValue:"
  Output: "0.77" (gamma^d as text)

Can also be trained with SLIME SFT loss mode.
"""

import json
import argparse
import os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)


def build_dataset(trajectory_path, tokenizer, max_length=512, oversample_positive=3):
    """Build HF dataset from trajectory JSONL.

    Labels are gamma^d values (0.0 for failed, 0.0 < x <= 1.0 for successful).
    Oversamples positive examples (label > 0) by the given factor.
    """
    samples = []
    positives = []

    with open(trajectory_path) as f:
        for line in f:
            d = json.loads(line)
            state = d["state"]  # Already formatted with prompt template
            label = d["label"]  # gamma^d value

            # Format as text completion: prompt already includes "Value: "
            completion = f"{label:.2f}"
            full_text = state + completion

            tokens = tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            sample = {
                "input_ids": tokens.input_ids.squeeze(0).tolist(),
                "attention_mask": tokens.attention_mask.squeeze(0).tolist(),
                "labels": tokens.input_ids.squeeze(0).tolist(),
            }
            samples.append(sample)

            if label > 0.0:
                positives.append(sample)

    # Oversample positives to handle class imbalance
    if oversample_positive > 1 and positives:
        for _ in range(int(oversample_positive) - 1):
            samples.extend(positives)

    return Dataset.from_list(samples)


def train(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    )

    dataset = build_dataset(
        args.data_path, tokenizer,
        max_length=args.max_length,
        oversample_positive=args.oversample_positive,
    )
    print(f"Training on {len(dataset)} samples (with oversampling)")

    split = dataset.train_test_split(test_size=0.1)

    training_args = TrainingArguments(
        output_dir=args.save_path,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        report_to="wandb" if args.wandb else "none",
        run_name="value-model-gamma-d",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()
    trainer.save_model(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    print(f"Value model saved to {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, help="Trajectory JSONL with gamma^d labels")
    parser.add_argument("--model-path", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--save-path", default="/mnt/filesystem-m5/formal/training/value_checkpoints")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--oversample-positive", type=float, default=3, help="Oversample proved states by this factor")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    train(args)
