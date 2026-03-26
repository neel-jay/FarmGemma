#!/usr/bin/env python3
"""
FarmGemma Data Pipeline
Data collection, preprocessing, and dataset creation for FarmGemma model.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from dataclasses import dataclass


@dataclass
class AgriculturalSample:
    """Single agricultural sample for training."""
    image_path: str = None
    text: str = None
    language: str = "en"
    category: str = None
    metadata: Dict = None


class AgriculturalDataset(Dataset):
    """PyTorch Dataset for agricultural data."""
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        languages: List[str] = ["en", "hi", "ta", "te"],
        max_samples: int = None
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.languages = languages
        self.samples = self._load_samples(max_samples)
    
    def _load_samples(self, max_samples: int = None) -> List[AgriculturalSample]:
        samples = []
        
        crop_diseases = self.data_dir / "crop_diseases"
        if crop_diseases.exists():
            samples.extend(self._load_crop_diseases(crop_diseases))
        
        pest_data = self.data_dir / "pest_identification"
        if pest_data.exists():
            samples.extend(self._load_pest_data(pest_data))
        
        qa_pairs = self.data_dir / "qa_pairs"
        if qa_pairs.exists():
            samples.extend(self._load_qa_pairs(qa_pairs))
        
        if max_samples:
            samples = samples[:max_samples]
        
        return samples
    
    def _load_crop_diseases(self, data_dir: Path) -> List[AgriculturalSample]:
        samples = []
        for disease_dir in data_dir.iterdir():
            if disease_dir.is_dir():
                for img_file in disease_dir.glob("*.jpg"):
                    samples.append(AgriculturalSample(
                        image_path=str(img_file),
                        text=f"Crop disease: {disease_dir.name}",
                        language="en",
                        category="crop_disease",
                        metadata={"disease_name": disease_dir.name}
                    ))
        return samples
    
    def _load_pest_data(self, data_dir: Path) -> List[AgriculturalSample]:
        samples = []
        for pest_dir in data_dir.iterdir():
            if pest_dir.is_dir():
                for img_file in pest_dir.glob("*.jpg"):
                    samples.append(AgriculturalSample(
                        image_path=str(img_file),
                        text=f"Pest: {pest_dir.name}",
                        language="en",
                        category="pest",
                        metadata={"pest_name": pest_dir.name}
                    ))
        return samples
    
    def _load_qa_pairs(self, data_dir: Path) -> List[AgriculturalSample]:
        samples = []
        for lang in self.languages:
            qa_file = data_dir / f"{lang}_qa.jsonl"
            if qa_file.exists():
                with open(qa_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line)
                        samples.append(AgriculturalSample(
                            text=data.get("question", "") + " " + data.get("answer", ""),
                            language=lang,
                            category="qa",
                            metadata=data
                        ))
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        item = {
            "text": sample.text,
            "language": sample.language,
            "category": sample.category,
            "metadata": sample.metadata
        }
        
        if sample.image_path and os.path.exists(sample.image_path):
            item["image"] = Image.open(sample.image_path).convert("RGB")
        
        return item


def create_knowledge_base_index(knowledge_dir: str, output_file: str):
    """Create an index of all knowledge base documents."""
    index = []
    knowledge_path = Path(knowledge_dir)
    
    for file_path in knowledge_path.rglob("*"):
        if file_path.is_file() and file_path.suffix in ['.txt', '.md', '.json', '.pdf']:
            index.append({
                "path": str(file_path),
                "name": file_path.name,
                "category": file_path.parent.name,
                "size": file_path.stat().st_size
            })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2)
    
    print(f"Knowledge base index created: {len(index)} documents")


def merge_qa_datasets(output_file: str, input_dirs: List[str]):
    """Merge Q&A datasets from multiple sources."""
    all_qa = []
    
    for input_dir in input_dirs:
        input_path = Path(input_dir)
        for qa_file in input_path.glob("**/*_qa.jsonl"):
            with open(qa_file, 'r', encoding='utf-8') as f:
                for line in f:
                    all_qa.append(json.loads(line))
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for qa in all_qa:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")
    
    print(f"Merged QA dataset: {len(all_qa)} pairs")


def main():
    parser = argparse.ArgumentParser(description="FarmGemma Data Pipeline")
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory")
    parser.add_argument("--action", type=str, required=True,
                        choices=["index_kb", "merge_qa", "create_dataset"])
    parser.add_argument("--output", type=str, help="Output file")
    parser.add_argument("--languages", type=str, nargs="+",
                        default=["en", "hi", "ta", "te", "mr", "bn", "kn", "gu", "pa"])
    
    args = parser.parse_args()
    
    if args.action == "index_kb":
        create_knowledge_base_index(args.data_dir, args.output or "kb_index.json")
    elif args.action == "merge_qa":
        merge_qa_datasets(args.output or "merged_qa.jsonl", [args.data_dir])
    elif args.action == "create_dataset":
        dataset = AgriculturalDataset(args.data_dir, languages=args.languages)
        print(f"Dataset created: {len(dataset)} samples")


if __name__ == "__main__":
    main()