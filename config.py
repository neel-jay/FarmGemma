# FarmGemma - AI Assistant for Indian Farmers
# Fine-tuned Gemma model for agriculture advisory

model_name = "farmgemma-4b"
base_model = "google/gemma-3-4b-it"
vision_encoder = "google/siglip-so400m-finetuned"

languages = ["en", "hi", "ta", "te", "mr", "bn", "kn", "gu", "pa"]

model_variants = {
    "farmgemma-1b": {
        "base": "google/gemma-3-1b-it",
        "params": "1B",
        "use_case": "Text-only Q&A",
        "device": "Entry-level smartphones"
    },
    "farmgemma-4b": {
        "base": "google/gemma-3-4b-it",
        "params": "4B",
        "use_case": "Multimodal (text + images)",
        "device": "Mid-range smartphones"
    },
    "farmgemma-4b-edge": {
        "base": "google/gemma-3-4b-it",
        "params": "4B",
        "quantized": "int4",
        "use_case": "Offline pest detection",
        "device": "Raspberry Pi, Kisan Drones"
    }
}

training_config = {
    "batch_size": 8,
    "learning_rate": 2e-5,
    "epochs": 3,
    "warmup_steps": 100,
    "max_seq_length": 2048,
    "vision_frame_size": 224
}

capabilities = [
    "crop_disease_detection",
    "pest_identification",
    "soil_analysis",
    "weather_advisory",
    "mandi_prices",
    "government_schemes",
    "voice_interface"
]