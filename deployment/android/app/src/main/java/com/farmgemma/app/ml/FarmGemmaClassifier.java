package com.farmgemma.app.ml;

import android.content.Context;
import android.graphics.Bitmap;
import org.tensorflow.lite.Interpreter;
import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class FarmGemmaClassifier {
    
    private final Interpreter interpreter;
    private final int inputSize = 224;
    private final int numThreads = 4;
    
    public FarmGemmaClassifier(Context context, String modelPath) {
        this.interpreter = new Interpreter(loadModelFile(context, modelPath));
    }
    
    private File loadModelFile(Context context, String modelPath) {
        return new File(context.getFilesDir(), modelPath);
    }
    
    public String classifyImage(Bitmap bitmap) {
        Bitmap resized = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true);
        ByteBuffer inputBuffer = convertBitmapToByteBuffer(resized);
        
        float[][] output = new float[1][10];
        interpreter.run(inputBuffer, output);
        
        return getDiseaseLabel(output[0]);
    }
    
    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer buffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3);
        buffer.order(ByteOrder.nativeOrder());
        
        int[] pixels = new int[inputSize * inputSize];
        bitmap.getPixels(pixels, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        
        for (int pixel : pixels) {
            buffer.putFloat((pixel >> 16 & 0xFF) / 255.0f);
            buffer.putFloat((pixel >> 8 & 0xFF) / 255.0f);
            buffer.putFloat((pixel & 0xFF) / 255.0f);
        }
        
        return buffer;
    }
    
    private String getDiseaseLabel(float[] probabilities) {
        String[] labels = {
            "rice_blast", "cotton_bollworm", "tomato_leaf_curl",
            "wheat_rust", "maize_stalk_rot", "healthy", "unknown"
        };
        
        int maxIdx = 0;
        float maxProb = probabilities[0];
        for (int i = 1; i < probabilities.length; i++) {
            if (probabilities[i] > maxProb) {
                maxProb = probabilities[i];
                maxIdx = i;
            }
        }
        
        return labels[maxIdx];
    }
}