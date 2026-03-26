#!/usr/bin/env python3
"""
FarmGemma Data Generator - Generate 50K+ Q&A pairs using Gemini
Run this on Colab to generate training data before training.
"""

import os
import json
import time
from typing import List, Dict

# Install Gemini if needed
try:
    import google.generativeai as genai
except ImportError:
    os.system("pip install -q google-generativeai")
    import google.generativeai as genai

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or input("Enter Gemini API Key: ")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# Language configurations
LANGUAGES = {
    "hi": {"name": "Hindi", "prompt_lang": "Hindi"},
    "ta": {"name": "Tamil", "prompt_lang": "Tamil"},
    "te": {"name": "Telugu", "prompt_lang": "Telugu"},
    "mr": {"name": "Marathi", "prompt_lang": "Marathi"},
    "bn": {"name": "Bengali", "prompt_lang": "Bengali"},
    "kn": {"name": "Kannada", "prompt_lang": "Kannada"},
    "gu": {"name": "Gujarati", "prompt_lang": "Gujarati"},
    "pa": {"name": "Punjabi", "prompt_lang": "Punjabi"},
    "en": {"name": "English", "prompt_lang": "English"}
}

# Topic templates for generating Q&A
TOPIC_TEMPLATES = {
    "crop_disease": [
        "What are the symptoms of {crop} {disease} and how to control it?",
        "My {crop} has {symptom}. What disease is this?",
        "How to prevent {crop} {disease} organically?",
        "What is the best fungicide for {crop} {disease}?",
        "Identify this {crop} disease: {description}",
    ],
    "pest_control": [
        "How to control {pest} in {crop}?",
        "Natural remedy for {pest} on {crop}?",
        "What pesticide to use for {pest} in {crop}?",
        "Signs of {pest} infestation in {crop}?",
        "Organic control of {pest} in {crop}?",
    ],
    "fertilizer": [
        "How much {fertilizer} to apply for {crop} per acre?",
        "What is NPK ratio for {crop}?",
        "When to apply {fertilizer} for {crop}?",
        "Signs of {nutrient} deficiency in {crop}?",
        "Organic fertilizer recommendation for {crop}?",
    ],
    "irrigation": [
        "How often to irrigate {crop}?",
        "Drip irrigation schedule for {crop}?",
        "Water requirement for {crop} per acre?",
        "Best time to irrigate {crop}?",
        "How to conserve water in {crop} farming?",
    ],
    "weather": [
        "Weather advisory for {crop} this week in {region}?",
        "Should I harvest {crop} before monsoon in {region}?",
        "What crops to sow in {month} in {region}?",
        "Frost advisory for {crop} in {region}?",
        "How does El Nino affect {crop} in {region}?",
    ],
    "schemes": [
        "How to apply for {scheme}?",
        "Eligibility for {scheme}?",
        "Documents needed for {scheme}?",
        "Benefits of {scheme} for farmers?",
        "Where to apply for {scheme} online?",
    ],
    "market": [
        "Today's price of {crop} at {mandi}?",
        "Best time to sell {crop} in {region}?",
        "Export demand for {crop}?",
        "Price trend of {crop} this season?",
        "How to get better price for {crop}?",
    ],
    "soil": [
        "How to improve {soil_type} soil for {crop}?",
        "pH level for {crop} cultivation?",
        "How to test soil at home?",
        "Soil preparation for {crop}?",
        "Compost making for {crop} farm?",
    ],
    "harvesting": [
        "When to harvest {crop}?",
        "Post-harvest management of {crop}?",
        "Storage tips for {crop}?",
        "How to reduce harvest losses in {crop}?",
        "Best method to thresh {crop}?",
    ]
}

# Entity values for template filling
CROPS = [
    "rice", "wheat", "cotton", "tomato", "potato", "onion", "garlic",
    "sugarcane", "maize", "bajra", "soybean", "groundnut", "mustard",
    "chickpea", "urad", "moong", "arhar", "banana", "mango", "grapes",
    "papaya", "okra", "brinjal", "chilli", "cabbage", "cauliflower"
]

DISEASES = [
    "blast", "blight", "rust", "rot", "curl", "spot", "mildew",
    "wilt", "canker", "scab", "anthracnose", "powdery mildew"
]

PESTS = [
    "stem borer", "leaf folder", "brown planthopper", "gall midge",
    "pink bollworm", "american bollworm", "whitefly", "aphid",
    "jassid", "thrips", "mealybug", "red spider mite", "fruit borer"
]

FERTILIZERS = [
    "urea", "DAP", "MOP", "SSP", "zinc sulphate", "boron",
    "gypsum", "lime", "neem cake", "bone meal", "vermicompost"
]

NUTRIENTS = [
    "nitrogen", "phosphorus", "potassium", "zinc", "iron",
    "manganese", "boron", "sulphur", "calcium", "magnesium"
]

REGIONS = [
    "Punjab", "Haryana", "Uttar Pradesh", "Madhya Pradesh",
    "Maharashtra", "Karnataka", "Tamil Nadu", "Andhra Pradesh",
    "Telangana", "West Bengal", "Gujarat", "Rajasthan"
]

MANDIS = [
    "Azadpur (Delhi)", "Vashi (Mumbai)", "Koyambedu (Chennai)",
    "Bowenpally (Hyderabad)", "Ghazipur (Delhi)", "Yeshwantpur (Bangalore)"
]

SCHEMES = [
    "PM-KISAN", "PMFBY", "Soil Health Card", "Kisan Credit Card",
    "PM Fasal Bima Yojana", "E-NAM", "Kisan Samman Nidhi"
]


def generate_qa_batch(topic: str, language: str, count: int = 50) -> List[Dict]:
    """Generate Q&A pairs for a topic in specified language."""
    
    lang_name = LANGUAGES[language]["name"]
    templates = TOPIC_TEMPLATES.get(topic, [])
    
    if not templates:
        return []
    
    prompt = f"""Generate {count} realistic agricultural Q&A pairs for Indian farmers in {lang_name}.

Topic: {topic}
Language: {lang_name}

Generate diverse questions covering different crops, regions, and scenarios.
Make answers practical, specific (with dosages, timings, variety names).

Format as JSON array:
[
  {{"question": "...", "answer": "..."}},
  ...
]

IMPORTANT: Questions should be in {lang_name} language.
IMPORTANT: Answers should be in {lang_name} language with specific recommendations.
"""

    try:
        response = model.generate_content(prompt)
        
        # Parse JSON from response
        text = response.text
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        qa_list = json.loads(text.strip())
        
        # Add metadata
        for qa in qa_list:
            qa["language"] = language
            qa["topic"] = topic
        
        return qa_list
    
    except Exception as e:
        print(f"Error generating {topic} in {language}: {e}")
        return []


def generate_all_data(output_file: str, samples_per_topic: int = 100):
    """Generate complete dataset."""
    
    all_qa = []
    topics = list(TOPIC_TEMPLATES.keys())
    languages = list(LANGUAGES.keys())
    
    total_generations = len(topics) * len(languages) * samples_per_topic
    print(f"🎯 Target: {total_generations:,} Q&A pairs")
    print(f"📝 Topics: {topics}")
    print(f"🌐 Languages: {list(LANGUAGES.keys())}")
    print()
    
    for topic in topics:
        for lang in languages:
            print(f"Generating {topic}/{lang}...", end=" ")
            
            qa_batch = generate_qa_batch(topic, lang, samples_per_topic)
            all_qa.extend(qa_batch)
            
            print(f"✓ {len(qa_batch)} pairs (total: {len(all_qa)})")
            
            # Rate limiting
            time.sleep(0.5)
    
    # Save
    print(f"\n💾 Saving {len(all_qa)} Q&A pairs to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for qa in all_qa:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")
    
    return all_qa


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate FarmGemma training data")
    parser.add_argument("--output", type=str, default="/content/drive/MyDrive/farmgemma/data/generated_qa.jsonl")
    parser.add_argument("--samples_per_topic", type=int, default=100,
                       help="Samples per topic per language (100 x 9 topics x 9 languages = 81,000)")
    parser.add_argument("--api_key", type=str, help="Gemini API Key")
    
    args = parser.parse_args()
    
    if args.api_key:
        GEMINI_API_KEY = args.api_key
        genai.configure(api_key=GEMINI_API_KEY)
    
    print("=" * 60)
    print("🌾 FarmGemma Data Generator 🌾")
    print("=" * 60)
    print()
    
    all_data = generate_all_data(args.output, args.samples_per_topic)
    
    print()
    print("=" * 60)
    print(f"✅ COMPLETE! Generated {len(all_data):,} Q&A pairs")
    print(f"📁 Saved to: {args.output}")
    print("=" * 60)