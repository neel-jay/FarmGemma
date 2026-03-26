package com.farmgemma.app;

import android.app.Application;
import org.tensorflow.lite.Interpreter;
import ai.onnxruntime.OnnxRuntime;
import java.io.File;

public class FarmGemmaApp extends Application {
    
    private Interpreter tfliteInterpreter;
    private OnnxRuntime onnxRuntime;
    
    @Override
    public void onCreate() {
        super.onCreate();
        initializeRuntime();
    }
    
    private void initializeRuntime() {
        // Initialize TensorFlow Lite for standard models
        try {
            tfliteInterpreter = new Interpreter(loadModelFile("farmgemma-4b.tflite"));
        } catch (Exception e) {
            tfliteInterpreter = null;
        }
        
        // Initialize ONNX Runtime for quantized edge models
        try {
            onnxRuntime = new OnnxRuntime();
        } catch (Exception e) {
            onnxRuntime = null;
        }
    }
    
    private File loadModelFile(String modelName) {
        return new File(getFilesDir(), modelName);
    }
    
    public Interpreter getTFLiteInterpreter() {
        return tfliteInterpreter;
    }
    
    public OnnxRuntime getOnnxRuntime() {
        return onnxRuntime;
    }
}