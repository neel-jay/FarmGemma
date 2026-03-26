# FarmGemma Dataset Structure

## Overview
This directory contains the training data structure for FarmGemma model.

```
data/
├── crop_diseases/           # 100K+ images with disease labels
│   ├── rice_blast/
│   ├── cotton_bollworm/
│   ├── tomato_leaf_curl/
│   └── ...
├── pest_identification/     # Insect pest images
│   ├── pink_bollworm/
│   ├── rice_weevil/
│   └── ...
├── soil_images/             # Soil type/quality images
│   ├── black_soil/
│   ├── red_soil/
│   └── ...
├── knowledge_base/          # Text corpus
│   ├── icar_research_papers/
│   ├── kisan_call_centre_logs/
│   ├── package_of_practices/
│   └── government_schemes/
├── qa_pairs/                # Multilingual Q&A
│   ├── en_qa.jsonl
│   ├── hi_qa.jsonl
│   └── ...
└── weather_crop_advisory/   # Location-specific advice
```

## Data Sources
- ICAR (Indian Council of Agricultural Research) publications
- Kisan Call Centre anonymized query logs
- Package of Practices from state agricultural universities
- Government scheme documentation (PM-KISAN, PMFBY, etc.)
- PlantVillage dataset
- IPM (Integrated Pest Management) image datasets

## Collection Partners
- ICAR
- State Agricultural Universities (SAUs)
- Krishi Vigyan Kendras (KVKs)
- Ministry of Agriculture & Farmers Welfare