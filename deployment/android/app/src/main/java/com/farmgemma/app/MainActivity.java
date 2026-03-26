package com.farmgemma.app;

import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import com.google.common.util.concurrent.ListenableFuture;
import java.io.File;
import java.text.SimpleDateFormat;
import java.util.concurrent.ExecutionException;

public class MainActivity extends AppCompatActivity {
    
    private static final String TAG = "FarmGemma";
    
    private ImageCapture imageCapture;
    private ImageView capturedImage;
    private TextView resultText;
    private FarmGemmaService farmGemmaService;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        
        initializeViews();
        initializeCamera();
        initializeFarmGemma();
    }
    
    private void initializeViews() {
        capturedImage = findViewById(R.id.capturedImage);
        resultText = findViewById(R.id.resultText);
        
        Button captureButton = findViewById(R.id.captureButton);
        captureButton.setOnClickListener(v -> captureImage());
        
        Button voiceButton = findViewById(R.id.voiceButton);
        voiceButton.setOnClickListener(v -> startVoiceInput());
    }
    
    private void initializeCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = 
                ProcessCameraProvider.getInstance(this);
        
        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                bindCameraUseCases(cameraProvider);
            } catch (ExecutionException | InterruptedException e) {
                Log.e(TAG, "Camera initialization failed", e);
            }
        }, ContextCompat.getMainExecutor(this));
    }
    
    private void bindCameraUseCases(ProcessCameraProvider cameraProvider) {
        Preview preview = new Preview.Builder().build();
        
        imageCapture = new ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build();
        
        CameraSelector cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA;
        
        try {
            cameraProvider.unbindAll();
            cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageCapture);
        } catch (Exception e) {
            Log.e(TAG, "Camera binding failed", e);
        }
    }
    
    private void initializeFarmGemma() {
        farmGemmaService = new FarmGemmaService(getApplicationContext());
    }
    
    private void captureImage() {
        if (imageCapture == null) return;
        
        File photoFile = new File(getFilesDir(), 
                "farmgemma_" + System.currentTimeMillis() + ".jpg");
        
        ImageCapture.OutputFileOptions outputOptions = 
                new ImageCapture.OutputFileOptions.Builder(photoFile).build();
        
        imageCapture.takePicture(outputOptions, 
                ContextCompat.getMainExecutor(this),
                new ImageCapture.OnImageSavedCallback() {
                    @Override
                    public void onImageSaved(@NonNull ImageCapture.OutputFileResults output) {
                        Bitmap bitmap = BitmapFactory.decodeFile(photoFile.getAbsolutePath());
                        capturedImage.setImageBitmap(bitmap);
                        analyzeImage(bitmap);
                    }
                    
                    @Override
                    public void onError(@NonNull ImageCaptureException exception) {
                        Log.e(TAG, "Image capture failed", exception);
                    }
                });
    }
    
    private void analyzeImage(Bitmap bitmap) {
        resultText.setText("Analyzing...");
        
        new Thread(() -> {
            String result = farmGemmaService.analyzeCropDisease(bitmap);
            runOnUiThread(() -> resultText.setText(result));
        }).start();
    }
    
    private void startVoiceInput() {
        Intent intent = new Intent(this, VoiceActivity.class);
        startActivity(intent);
    }
}