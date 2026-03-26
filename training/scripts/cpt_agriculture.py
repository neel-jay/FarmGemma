#!/usr/bin/env python3
"""
FarmGemma Continuous Pre-training Script
Pre-train Gemma on agricultural corpus for domain adaptation.
"""

import os
import torch
import argparse
from transformers import (
    GemmaForCausalLM,
    GemmaTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset


def setup_model(model_name: str = "google/gemma-3-4b"):
    """Initialize Gemma model for CPT."""
    model = GemmaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = GemmaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def prepare_cpt_dataset(data_dir: str, tokenizer, max_length: int = 2048):
    """Prepare dataset for continuous pre-training."""
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )
    
    dataset = load_dataset("text", data_dir=data_dir, split="train")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
    
    return tokenized_dataset


def train_cpt(
    model_name: str,
    data_dir: str,
    output_dir: str,
    epochs: int = 2,
    batch_size: int = 8,
    learning_rate: float = 2e-5
):
    """Run continuous pre-training on agricultural corpus."""
    
    print(f"Loading model: {model_name}")
    model, tokenizer = setup_model(model_name)
    
    print(f"Preparing dataset from: {data_dir}")
    dataset = prepare_cpt_dataset(data_dir, tokenizer)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        warmup_steps=100,
        logging_steps=50,
        save_steps=1000,
        bf16=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    print("Starting CPT training...")
    trainer.train()
    
    print(f"Saving model to: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


def main():
    parser = argparse.ArgumentParser(description="FarmGemma CPT Training")
    parser.add_argument("--model_name", type=str, default="google/gemma-3-4b")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    
    args = parser.parse_args()
    
    train_cpt(
        model_name=args.model_name,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )


if __name__ == "__main__":
    main()