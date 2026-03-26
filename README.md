# FarmGemma - AI Assistant for Indian Farmers

Fine-tuned Gemma model for agricultural advisory in India.

## FREE Training Options (No GPU needed locally!)

| Platform | GPU | Hours | Best For |
|----------|-----|-------|----------|
| **Google Colab** | T4 16GB | Unlimited | Quick prototyping |
| **Kaggle** | P100 16GB | 30-40/week | Longer training |
| **GCP** | T4/A100 | $300 free | Large models |

### Quick Start on Colab (Recommended):
1. Open `training/notebooks/colab_farmgemma_training.ipynb` in Google Colab
2. Runtime → Change runtime → **T4 GPU**
3. Run cells - that's it!

### Quick Start on Kaggle:
1. Go to kaggle.com → Create Notebook
2. Upload `training/notebooks/kaggle_farmgemma_training.ipynb`
3. Settings → Accelerator → **GPU P100**
4. Run cells

See `docs/FREE_CLOUD_TRAINING.md` for detailed setup instructions.

---

## Project Structure

## Project Structure

```
farmgemma/
├── config.py                 # Model configurations
├── data/                     # Dataset structure
│   ├── crop_diseases/
│   ├── pest_identification/
│   ├── soil_images/
│   ├── knowledge_base/
│   ├── qa_pairs/
│   └── weather_crop_advisory/
├── models/                   # Trained model checkpoints
│   ├── 1b/
│   ├── 4b/
│   └── 4b_edge/
├── training/
│   ├── notebooks/
│   │   └── farmgemma_finetuning.ipynb
│   ├── scripts/
│   │   ├── data_pipeline.py
│   │   ├── synthetic_data_generator.py
│   │   ├── cpt_agriculture.py
│   │   ├── fine_tune_vision.py
│   │   └── sft_multilingual.py
│   └── configs/
├── deployment/
│   ├── android/              # Android app (TFLite/ONNX)
│   ├── ios/                  # iOS app (CoreML)
│   ├── whatsapp/             # WhatsApp bot
│   └── ivr/                  # IVR system
└── evaluation/               # Evaluation framework
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Synthetic Training Data

```bash
python training/scripts/synthetic_data_generator.py \
    --api_key YOUR_GEMINI_API_KEY \
    --language hi \
    --category crop_disease \
    --output_dir data/qa_pairs
```

### 3. Run Data Pipeline

```bash
python training/scripts/data_pipeline.py \
    --data_dir data \
    --action create_dataset
```

### 4. Train Model

```bash
# Continuous Pre-training
python training/scripts/cpt_agriculture.py \
    --model_name google/gemma-3-4b \
    --data_dir data/knowledge_base \
    --output_dir models/farmgemma-4b-cpt

# Vision Fine-tuning
python training/scripts/fine_tune_vision.py \
    --vision_model google/siglip-so400m \
    --train_data data/crop_diseases \
    --output_dir models/farmsiglip

# SFT Training
python training/scripts/sft_multilingual.py \
    --model_path models/farmgemma-4b-cpt \
    --data_dir data/qa_pairs \
    --languages en hi ta te mr bn kn gu pa
```

## Model Variants

| Variant | Parameters | Use Case | Device |
|---------|------------|----------|--------|
| FarmGemma 1B | 1B | Text-only Q&A | Entry-level phones |
| FarmGemma 4B | 4B | Multimodal | Mid-range phones |
| FarmGemma 4B-Edge | 4B (INT4) | Offline detection | Raspberry Pi, Drones |

## Languages Supported

English, Hindi, Tamil, Telugu, Marathi, Bengali, Kannada, Gujarati, Punjabi

## Key Features

- Crop disease detection from photos
- Pest identification
- Weather-based advisory
- Mandi prices
- Government schemes information
- Voice interface (IVR)
- Offline capability

## Deployment

### Android
```bash
cd deployment/android
./gradlew assembleRelease
```

### WhatsApp Bot
```bash
cd deployment/whatsapp
python bot.py
```

## Evaluation

```bash
python evaluation/evaluator.py \
    --model_path models/farmgemma-4b-sft \
    --test_data evaluation/test_data/ \
    --output evaluation_results.json
```