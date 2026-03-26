#!/usr/bin/env python3
"""
FarmGemma Evaluation Framework
Evaluate model performance on agricultural advisory tasks.
"""

import json
import argparse
from typing import List, Dict, Any
from dataclasses import dataclass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sklearn.metrics import precision_recall_f1_support, accuracy_score


@dataclass
class EvaluationResult:
    """Result of model evaluation."""
    metric: str
    score: float
    details: Dict[str, Any] = None


class FarmGemmaEvaluator:
    """Evaluate FarmGemma model performance."""
    
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=256
        )
    
    def evaluate_disease_detection(self, test_data: List[Dict]) -> EvaluationResult:
        """Evaluate crop disease detection accuracy."""
        
        correct = 0
        total = len(test_data)
        
        predictions = []
        references = []
        
        for sample in test_data:
            image_desc = sample["image_description"]
            true_disease = sample["disease"]
            
            prompt = f"""Identify the crop disease from this description: {image_desc}

Disease:"""
            
            response = self.pipeline(prompt)[0]["generated_text"]
            predicted = self._extract_disease(response)
            
            predictions.append(predicted)
            references.append(true_disease)
            
            if predicted.lower() == true_disease.lower():
                correct += 1
        
        accuracy = accuracy_score(references, predictions)
        precision, recall, f1, _ = precision_recall_f1_support(
            references, predictions, average="macro", zero_division=0
        )
        
        return EvaluationResult(
            metric="disease_detection",
            score=accuracy,
            details={
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "correct": correct,
                "total": total
            }
        )
    
    def evaluate_qa_relevance(self, test_data: List[Dict]) -> EvaluationResult:
        """Evaluate Q&A response relevance."""
        
        relevance_scores = []
        
        for sample in test_data:
            question = sample["question"]
            true_answer = sample["answer"]
            
            prompt = f"""Question: {question}

Answer:"""
            
            generated = self.pipeline(prompt)[0]["generated_text"]
            generated = generated.split("Answer:")[-1].strip()
            
            relevance = self._calculate_relevance(generated, true_answer)
            relevance_scores.append(relevance)
        
        avg_relevance = sum(relevance_scores) / len(relevance_scores)
        
        return EvaluationResult(
            metric="qa_relevance",
            score=avg_relevance,
            details={"scores": relevance_scores}
        )
    
    def evaluate_multilingual(self, test_data: List[Dict]) -> EvaluationResult:
        """Evaluate multilingual capability."""
        
        language_scores = {}
        
        for sample in test_data:
            language = sample["language"]
            question = sample["question"]
            
            if language not in language_scores:
                language_scores[language] = []
            
            prompt = f"""Answer this agricultural question in {language}:
            
Question: {question}

Answer ({language}):"""
            
            generated = self.pipeline(prompt)[0]["generated_text"]
            fluency_score = self._assess_fluency(generated, language)
            language_scores[language].append(fluency_score)
        
        avg_scores = {
            lang: sum(scores) / len(scores)
            for lang, scores in language_scores.items()
        }
        
        return EvaluationResult(
            metric="multilingual",
            score=sum(avg_scores.values()) / len(avg_scores),
            details={"per_language": avg_scores}
        )
    
    def _extract_disease(self, text: str) -> str:
        """Extract disease name from response."""
        text = text.lower()
        diseases = ["blast", "rust", "blight", "rot", "curl", "spot", "mildew"]
        for disease in diseases:
            if disease in text:
                return disease
        return "unknown"
    
    def _calculate_relevance(self, generated: str, reference: str) -> float:
        """Calculate relevance score between generated and reference."""
        gen_words = set(generated.lower().split())
        ref_words = set(reference.lower().split())
        
        if not ref_words:
            return 0.0
        
        overlap = len(gen_words.intersection(ref_words))
        return overlap / len(ref_words)
    
    def _assess_fluency(self, text: str, language: str) -> float:
        """Assess language fluency of generated text."""
        words = text.split()
        
        if len(words) < 5:
            return 0.0
        
        min_length = 10 if language == "en" else 5
        if len(words) >= min_length:
            return 1.0
        
        return len(words) / min_length


def main():
    parser = argparse.ArgumentParser(description="FarmGemma Evaluation")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output", type=str, default="evaluation_results.json")
    
    args = parser.parse_args()
    
    evaluator = FarmGemmaEvaluator(args.model_path)
    
    with open(args.test_data, 'r') as f:
        test_data = json.load(f)
    
    results = []
    
    if "crop_diseases" in test_data:
        result = evaluator.evaluate_disease_detection(test_data["crop_diseases"])
        results.append(result)
        print(f"Disease Detection Accuracy: {result.score:.4f}")
    
    if "qa_pairs" in test_data:
        result = evaluator.evaluate_qa_relevance(test_data["qa_pairs"])
        results.append(result)
        print(f"Q&A Relevance Score: {result.score:.4f}")
    
    if "multilingual" in test_data:
        result = evaluator.evaluate_multilingual(test_data["multilingual"])
        results.append(result)
        print(f"Multilingual Score: {result.score:.4f}")
    
    output = [
        {
            "metric": r.metric,
            "score": r.score,
            "details": r.details
        }
        for r in results
    ]
    
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()