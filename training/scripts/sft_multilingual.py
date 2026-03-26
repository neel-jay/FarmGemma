#!/usr/bin/env python3
"""
FarmGemma Multilingual SFT Training
Instruction fine-tuning on multilingual agricultural Q&A.
"""

import torch
import json
import argparse
from transformers import (
    GemmaForCausalLM,
    GemmaTokenizer,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset


def prepare_sft_dataset(data_dir: str, tokenizer, languages: list, max_length: int = 2048):
    """Prepare dataset for SFT."""
    
    def format_sample(sample, language):
        prompt = f"""You are FarmGemma, an AI assistant for Indian farmers.

Question: {sample['question']}

Answer: {sample['answer']}"""
        return tokenizer(prompt, truncation=True, max_length=max_length, padding="max_length")
    
    all_samples = []
    
    for lang in languages:
        lang_file = f"{data_dir}/{lang}_qa.jsonl"
        if not lang_file:
            continue
            
        with open(lang_file, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line)
                all_samples.append(format_sample(sample, lang))
    
    return all_samples


def train_sft(
    model_path: str,
    data_dir: str,
    output_dir: str,
    languages: list,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5
):
    """Run SFT training on multilingual Q&A data."""
    
    print(f"Loading model from: {model_path}")
    model = GemmaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = GemmaTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Preparing SFT dataset for languages: {languages}")
    dataset = prepare_sft_dataset(data_dir, tokenizer, languages)
    
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
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    
    print("Starting SFT training...")
    trainer.train()
    
    print(f"Saving model to: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


def main():
    parser = argparse.ArgumentParser(description="FarmGemma SFT Training")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--languages", type=str, nargs="+",
                        default=["en", "hi", "ta", "te", "mr", "bn", "kn", "gu", "pa"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    
    args = parser.parse_args()
    
    train_sft(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        languages=args.languages,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )


if __name__ == "__main__":
    main()