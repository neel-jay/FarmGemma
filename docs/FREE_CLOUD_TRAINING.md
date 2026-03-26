# Free Cloud Training Options for FarmGemma

## 1. Google Colab (RECOMMENDED - Easiest)

**GPU**: T4 16GB (Free)
**Best for**: Quick experiments, prototyping

### Steps:
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Upload `colab_farmgemma_training.ipynb`
3. Runtime → Change runtime → T4 GPU
4. Run cells

**Limits**: ~12-24 hours continuous, disconnects when idle

---

## 2. Kaggle Notebooks (Better for longer training)

**GPU**: P100 16GB (Free, 30-40 hrs/week)
**Best for**: Longer training sessions

### Steps:
1. Go to [kaggle.com](https://kaggle.com)
2. Create New Notebook
3. Upload `kaggle_farmgemma_training.ipynb`
4. Settings → Accelerator → GPU P100
5. Internet → On

**Limits**: 30-40 hours per week of GPU

---

## 3. Google Cloud Platform ($300 Free Credits)

**GPU**: A100, T4 (Free tier)
**Best for**: Heavy training, larger models

### Setup:
```bash
# 1. Get $300 credits at https://cloud.google.com/free
# 2. Create a new project
# 3. Enable TensorFlow or PyTorch VM instance

# Create VM with T4 (free tier eligible)
gcloud compute instances create farmgemma-training \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=pytorch-latest \
    --image-project=deep-learning-platform \
    --boot-disk-size=100GB \
    --scopes=cloud-platform

# SSH into VM and run training
gcloud compute ssh farmgemma-training --zone=us-central1-a
```

**Limits**: $300 credits for 90 days, T4 limited

---

## 4. Lambda Labs (Free GPU Hours)

**GPU**: T4, RTX 4000 (Free hours)
**Best for**: Hands-off training

### Steps:
1. Sign up at [lambdalabs.com](https://lambdalabs.com)
2. Get free 4 hours of GPU
3. Use Jupyter notebook or SSH

---

## 5. Paperspace Gradient (Free Tier)

**GPU**: P4000 (Free)
**Best for**: ML workflows

### Steps:
1. Sign up at [paperspace.com](https://paperspace.com)
2. Create Gradient notebook
3. Free P4000 GPU available

---

## Recommended Training Strategy

### Week 1: Prototype on Colab
- Train Gemma 1B on synthetic Q&A data
- Verify training pipeline works
- Save checkpoints to Google Drive

### Week 2: Scale on Kaggle
- Use P100 for longer training
- Train on larger dataset
- Download final model

### Week 3: Optional GCP for Large Model
- Use $300 credits for Gemma 4B
- Train with full dataset

---

## Dataset Sources (All Free & Public)

| Dataset | Source | Description |
|---------|--------|-------------|
| PlantVillage | HuggingFace | 54K crop disease images |
| Agri-QA | HuggingFace | Agricultural Q&A pairs |
| CropNet | TensorFlow Hub | Plant disease classification |
| iNaturalist | AWS Open | 5M+ species images |
| ICAR Data | Direct | Government ag data |

---

## Quick Start Commands

```bash
# Clone repo
git clone https://github.com/your-repo/farmgemma.git
cd farmgemma

# For Colab - just upload notebooks/colab_farmgemma_training.ipynb

# For Kaggle - upload notebooks/kaggle_farmgemma_training.ipynb

# For GCP VM
gcloud compute ssh farmgemma-training --zone=us-central1-a
python training/scripts/sft_multilingual.py \
    --model_path google/gemma-3-1b-it \
    --data_dir ./data/qa_pairs \
    --output_dir ./outputs
```