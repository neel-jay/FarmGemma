package com.farmgemma.app;

import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import android.speech.SpeechRecognizer;
import android.Manifest;
import android.content.pm.PackageManager;
import androidx.annotation.NonNull;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import java.util.Locale;

public class VoiceActivity extends AppCompatActivity {
    
    private static final int SPEECH_REQUEST_CODE = 100;
    private static final int RECORD_AUDIO_REQUEST_CODE = 101;
    
    private TextToSpeech tts;
    private SpeechRecognizer speechRecognizer;
    private String currentLanguage = "hi_IN";
    
    private final HashMap<String, Locale> languageLocales = new HashMap() {{
        put("hi", new Locale("hi", "IN"));
        put("ta", new Locale("ta", "IN"));
        put("te", new Locale("te", "IN"));
        put("mr", new Locale("mr", "IN"));
        put("bn", new Locale("bn", "IN"));
        put("kn", new Locale("kn", "IN"));
        put("gu", new Locale("gu", "IN"));
        put("pa", new Locale("pa", "IN"));
        put("en", Locale.ENGLISH);
    }};
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        
        initializeTTS();
        requestAudioPermissions();
    }
    
    private void initializeTTS() {
        tts = new TextToSpeech(this, status -> {
            if (status == TextToSpeech.SUCCESS) {
                tts.setLanguage(languageLocales.get(currentLanguage));
            }
        });
    }
    
    private void requestAudioPermissions() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) 
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                new String[]{Manifest.permission.RECORD_AUDIO},
                RECORD_AUDIO_REQUEST_CODE);
        }
    }
    
    public void speakText(String text) {
        if (tts != null) {
            tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, "farmgemma_tts");
        }
    }
    
    public void startListening() {
        Intent intent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
        intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE, currentLanguage);
        intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, 
                RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
        startActivityForResult(intent, SPEECH_REQUEST_CODE);
    }
    
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @NonNull Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == SPEECH_REQUEST_CODE && resultCode == RESULT_OK) {
            ArrayList<String> results = data.getStringArrayListExtra(
                    RecognizerIntent.EXTRA_RESULTS);
            if (results != null && !results.isEmpty()) {
                String spokenText = results.get(0);
                processVoiceInput(spokenText);
            }
        }
    }
    
    private void processVoiceInput(String text) {
        // Send to FarmGemma model for processing
        String response = farmGemmaService.query(text, currentLanguage);
        speakText(response);
    }
}