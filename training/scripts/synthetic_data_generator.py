#!/usr/bin/env python3
"""
FarmGemma Synthetic Dataset Generator
Generate synthetic agricultural Q&A pairs using Gemini API to bootstrap training data.
"""

import os
import json
import argparse
from typing import List, Dict
from dataclasses import dataclass
import google.generativeai as genai


@dataclass
class QAPair:
    """Agricultural Q&A pair."""
    question: str
    answer: str
    language: str
    category: str
    region: str = None
    crop: str = None


LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "bn": "Bengali",
    "kn": "Kannada",
    "gu": "Gujarati",
    "pa": "Punjabi"
}

CATEGORIES = [
    "crop_disease",
    "pest_identification", 
    "soil_management",
    "weather_advisory",
    "mandi_prices",
    "government_schemes",
    "fertilizer_recommendation",
    "irrigation",
    "harvesting"
]

CROPS = [
    "rice", "wheat", "cotton", "tomato", "potato", "onion", "sugarcane",
    "maize", "soybean", "groundnut", "mustard", "chickpea", "banana",
    "mango", "grapes", "apple", "tea", "coffee"
]

REGIONS = [
    "Punjab", "Haryana", "Uttar Pradesh", "Maharashtra", "Karnataka",
    "Tamil Nadu", "Andhra Pradesh", "West Bengal", "Gujarat", "Rajasthan"
]


class SyntheticDatasetGenerator:
    """Generate synthetic agricultural Q&A data."""
    
    def __init__(self, api_key: str = None):
        if api_key:
            genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
    
    def generate_qa_pairs(
        self,
        language: str,
        category: str,
        crop: str = None,
        count: int = 5
    ) -> List[QAPair]:
        """Generate Q&A pairs for a specific category and language."""
        
        lang_name = LANGUAGES.get(language, "English")
        
        prompt = f"""Generate {count} realistic agricultural Q&A pairs in {lang_name} for Indian farmers.

Category: {category}
Crop: {crop if crop else 'various crops'}
Language: {lang_name}

Format each Q&A as JSON:
{{"question": "...", "answer": "..."}}

Make questions realistic and answers practical and actionable.
Include specific product names, dosages, and local context where applicable."""

        try:
            response = self.model.generate_content(prompt)
            
            qa_pairs = []
            for line in response.text.strip().split('\n'):
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    try:
                        data = json.loads(line)
                        qa_pairs.append(QAPair(
                            question=data.get("question", ""),
                            answer=data.get("answer", ""),
                            language=language,
                            category=category,
                            crop=crop,
                            region=REGIONS[0] if crop else None
                        ))
                    except json.JSONDecodeError:
                        continue
            
            return qa_pairs
        
        except Exception as e:
            print(f"Error generating Q&A pairs: {e}")
            return []
    
    def generate_crop_disease_dataset(self, language: str, crop: str) -> List[QAPair]:
        """Generate crop disease Q&A for a specific crop."""
        return self.generate_qa_pairs(
            language=language,
            category="crop_disease",
            crop=crop,
            count=10
        )
    
    def generate_pest_dataset(self, language: str, pest_type: str = None) -> List[QAPair]:
        """Generate pest identification Q&A."""
        return self.generate_qa_pairs(
            language=language,
            category="pest_identification",
            count=10
        )
    
    def generate_scheme_dataset(self, language: str) -> List[QAPair]:
        """Generate government schemes Q&A."""
        return self.generate_qa_pairs(
            language=language,
            category="government_schemes",
            count=10
        )


def save_qa_pairs(qa_pairs: List[QAPair], output_file: str):
    """Save Q&A pairs to JSONL file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for qa in qa_pairs:
            f.write(json.dumps({
                "question": qa.question,
                "answer": qa.answer,
                "language": qa.language,
                "category": qa.category,
                "crop": qa.crop,
                "region": qa.region
            }, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="FarmGemma Synthetic Dataset Generator")
    parser.add_argument("--api_key", type=str, help="Gemini API key")
    parser.add_argument("--language", type=str, default="en", choices=list(LANGUAGES.keys()))
    parser.add_argument("--category", type=str, default="all", 
                        choices=["all"] + CATEGORIES)
    parser.add_argument("--output_dir", type=str, default="./data/qa_pairs")
    parser.add_argument("--count", type=int, default=50, help="Number of Q&A pairs per category")
    
    args = parser.parse_args()
    
    generator = SyntheticDatasetGenerator(api_key=args.api_key or os.getenv("GEMINI_API_KEY"))
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    categories = CATEGORIES if args.category == "all" else [args.category]
    
    for category in categories:
        print(f"Generating {args.language} Q&A for category: {category}")
        qa_pairs = generator.generate_qa_pairs(
            language=args.language,
            category=category,
            count=args.count
        )
        
        if qa_pairs:
            output_file = output_dir / f"{args.language}_{category}_qa.jsonl"
            save_qa_pairs(qa_pairs, str(output_file))
            print(f"  Saved {len(qa_pairs)} pairs to {output_file}")


if __name__ == "__main__":
    main()