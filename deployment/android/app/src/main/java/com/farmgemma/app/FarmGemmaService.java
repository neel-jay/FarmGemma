package com.farmgemma.app;

import android.content.Context;
import android.graphics.Bitmap;
import java.io.File;

public class FarmGemmaService {
    
    private final FarmGemmaClassifier classifier;
    private final FarmGemmaLLM llm;
    private final Context context;
    
    public FarmGemmaService(Context context) {
        this.context = context;
        this.classifier = new FarmGemmaClassifier(context, "farmgemma-disease.tflite");
        this.llm = new FarmGemmaLLM(context, "farmgemma-4b-quantized.onnx");
    }
    
    public String analyzeCropDisease(Bitmap bitmap) {
        String diseaseLabel = classifier.classifyImage(bitmap);
        String advisory = llm.query(
            "What is " + diseaseLabel + " and how to treat it?",
            "en"
        );
        return diseaseLabel + "\n\n" + advisory;
    }
    
    public String query(String question, String language) {
        return llm.query(question, language);
    }
    
    public String getWeatherAdvisory(String location) {
        return llm.query(
            "What is the weather forecast for " + location + " and its impact on farming?",
            "en"
        );
    }
    
    public String getMandiPrice(String crop, String market) {
        return llm.query(
            "What is the current price of " + crop + " at " + market + " mandi?",
            "en"
        );
    }
}