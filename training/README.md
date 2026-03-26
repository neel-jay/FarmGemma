# FarmGemma Training Pipeline

## Overview
Multi-stage training pipeline for FarmGemma model.

### Stage 1: Continuous Pre-training (CPT)
```bash
# Train on agriculture corpus
python training/scripts/cpt_agriculture.py \
    --base_model google/gemma-3-4b \
    --corpus data/knowledge_base \
    --output models/farmgemma-4b-cpt \
    --epochs 2 \
    --batch_size 16
```

### Stage 2: Vision Encoder Fine-tuning (FarmSigLIP)
```bash
# Fine-tune SigLIP on crop/pest images
python training/scripts/fine_tune_vision.py \
    --base_model google/siglip-so400m \
    --image_data data/crop_diseases \
    --pest_data data/pest_identification \
    --output models/farmsiglip
```

### Stage 3: Multimodal Alignment
```bash
# Pair crop images with advisory text
python training/scripts/align_multimodal.py \
    --vision_encoder models/farmsiglip \
    --llm models/farmgemma-4b-cpt \
    --image_text_pairs data/qa_pairs \
    --output models/farmgemma-4b-aligned
```

### Stage 4: Instruction Fine-tuning (SFT)
```bash
# Train on multilingual Q&A pairs
python training/scripts/sft_multilingual.py \
    --model models/farmgemma-4b-aligned \
    --qa_data data/qa_pairs \
    --languages hi ta te mr bn kn gu pa en \
    --output models/farmgemma-4b-sft
```

### Stage 5: RLHF with Farmer Feedback
```bash
# Human feedback from agricultural extension officers
python training/scripts/rlhf.py \
    --sft_model models/farmgemma-4b-sft \
    --feedback_data evaluation/farmer_feedback \
    --output models/farmgemma-4b-rlhf
```

## Quantization for Edge Deployment
```bash
# Quantize to INT4 for edge devices
python training/scripts/quantize.py \
    --model models/farmgemma-4b-rlhf \
    --quantization int4 \
    --output models/farmgemma-4b-edge
```