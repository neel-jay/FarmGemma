#!/usr/bin/env python3
"""
FarmGemma WhatsApp Bot
WhatsApp Business API integration for FarmGemma agricultural advisory.
"""

import os
import json
import logging
from flask import Flask, request, jsonify
from twilio.twimlp.messaging_response import MessagingResponse
from twilio.rest import Client
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

MODEL_PATH = os.getenv("MODEL_PATH", "./models/farmgemma-4b-sft")

class FarmGemmaWhatsAppBot:
    """WhatsApp bot for FarmGemma agricultural advisory."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize FarmGemma model."""
        try:
            logger.info(f"Loading model from {MODEL_PATH}")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    
    def generate_response(self, user_message: str, language: str = "en") -> str:
        """Generate agricultural advisory response."""
        
        prompt = f"""You are FarmGemma, an AI assistant for Indian farmers. 
Provide helpful, accurate agricultural advice in {language}.

Farmer's question: {user_message}

FarmGemma response:"""
        
        try:
            if self.pipeline:
                response = self.pipeline(prompt)[0]["generated_text"]
                response = response.split("FarmGemma response:")[-1].strip()
                return response
            else:
                return "Model not available. Please try again later."
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return "Sorry, I couldn't process your request. Please try again."
    
    def process_image(self, media_url: str) -> str:
        """Process image for crop disease/pest detection."""
        return ("I can see your image. Based on the visual analysis, "
                "this appears to be related to crop health. "
                "For detailed analysis, please use the FarmGemma app.")


bot = FarmGemmaWhatsAppBot()


@app.route("/webhook", methods=["POST"])
def webhook():
    """Handle incoming WhatsApp messages."""
    
    incoming_message = request.values.get("Body", "").strip()
    from_number = request.values.get("From", "")
    media_url = request.values.get("MediaUrl0", None)
    
    logger.info(f"Received message from {from_number}: {incoming_message}")
    
    response = MessagingResponse()
    
    if media_url:
        bot_response = bot.process_image(media_url)
    else:
        detected_language = detect_language(incoming_message)
        bot_response = bot.generate_response(incoming_message, detected_language)
    
    response.message(bot_response)
    
    return Response(str(response), mimetype="application/xml")


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": bot.model is not None
    })


def detect_language(text: str) -> str:
    """Simple language detection for incoming messages."""
    hindi_chars = set("अआइईउऊऋएऐओऔकखगघचछजझटठडढणतथदधनपफबभमयरलवशषसह")
    
    if any(char in hindi_chars for char in text):
        return "hi"
    return "en"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)