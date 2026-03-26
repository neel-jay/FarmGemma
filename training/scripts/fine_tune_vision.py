#!/usr/bin/env python3
"""
FarmGemma Vision Fine-tuning
Fine-tune SigLIP vision encoder on crop/pest images.
"""

import torch
import argparse
from transformers import AutoModel, AutoProcessor
from PIL import Image
import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


class CropPestDataset(Dataset):
    """Dataset for crop disease and pest images."""
    
    def __init__(self, image_dir: str, processor, max_samples: int = None):
        self.image_dir = Path(image_dir)
        self.processor = processor
        self.samples = []
        
        for category_dir in self.image_dir.iterdir():
            if category_dir.is_dir():
                label = category_dir.name
                for img_path in category_dir.glob("*.jpg"):
                    self.samples.append((str(img_path), label))
                    if max_samples and len(self.samples) >= max_samples:
                        break
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        
        inputs = self.processor(images=image, return_tensors="pt")
        inputs["labels"] = torch.tensor(self.label_to_id(label))
        
        return inputs


def fine_tune_vision(
    vision_model: str,
    train_data: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 1e-4
):
    """Fine-tune vision encoder on agricultural images."""
    
    print(f"Loading vision model: {vision_model}")
    model = AutoModel.from_pretrained(vision_model, torch_dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(vision_model)
    
    dataset = CropPestDataset(train_data, processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()
    
    print(f"Starting vision fine-tuning for {epochs} epochs...")
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            pixel_values = batch["pixel_values"].squeeze(1).to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    print(f"Saving vision model to: {output_dir}")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)


def main():
    parser = argparse.ArgumentParser(description="FarmGemma Vision Fine-tuning")
    parser.add_argument("--vision_model", type=str, default="google/siglip-so400m")
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    
    args = parser.parse_args()
    
    fine_tune_vision(
        vision_model=args.vision_model,
        train_data=args.train_data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )


if __name__ == "__main__":
    main()