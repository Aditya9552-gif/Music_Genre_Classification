import streamlit as st
import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from collections import Counter
import time # Import the time module for timing

# Load saved models
# Ensure these model files ('ann_feature_10sec_model.keras', etc.) are in the same directory
# as your Streamlit app or provide full paths.
try:
    feature_model1 = load_model('ann_feature_model.keras')
    feature_model2 = load_model('cnn_feature_model.keras')
    spectrogram_model1 = load_model('CNN_spectrogram_model.keras')
    spectrogram_model2 = load_model('GRU_spectrogram_model.keras')
    spectrogram_model3 = load_model('LSTM_spectrogram_model.keras')
    print("All models loaded successfully.") # For console feedback
except Exception as e:
    st.error(f"Error loading one or more models. Please ensure all .keras files are in the correct directory. Error: {e}")
    st.stop() # Stop the app if models can't be loaded

# Load scaler and label encoder for feature-based models
try:
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    print("Scaler and Label Encoder loaded successfully.") # For console feedback
except Exception as e:
    st.error(f"Error loading scaler or label encoder. Please ensure 'scaler.pkl' and 'label_encoder.pkl' are in the correct directory. Error: {e}")
    st.stop() # Stop the app if these essential files can't be loaded

# Function to extract features from audio
def extract_features_segment(audio_data, sr):
    chroma_stft = librosa.feature.chroma_stft(y=audio_data, sr=sr)
    rms = librosa.feature.rms(y=audio_data)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)
    harmony = librosa.effects.harmonic(y=audio_data)
    perceptr = librosa.effects.percussive(y=audio_data)
    
    # Fix: Extract the scalar from the tempo array
    tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sr)
    tempo = tempo.item() # Use .item() to get the scalar value, safe for 0-dim arrays

    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=20)
    
    # Extract mean & variance of each feature
    features = [
        np.mean(chroma_stft), np.var(chroma_stft),
        np.mean(rms), np.var(rms),
        np.mean(spectral_centroid), np.var(spectral_centroid),
        np.mean(spectral_bandwidth), np.var(spectral_bandwidth),
        np.mean(rolloff), np.var(rolloff),
        np.mean(zero_crossing_rate), np.var(zero_crossing_rate),
        np.mean(harmony), np.var(harmony),
        np.mean(perceptr), np.var(perceptr),
        tempo
    ] + list(np.mean(mfccs, axis=1)) + list(np.var(mfccs, axis=1))
    return features

# Function to create spectrogram from audio
def create_spectrogram(audio_data, sr, temp_dir, input_shape):
    plt.figure(figsize=(3, 3))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', cmap='viridis')
    plt.axis('off')
    plt.tight_layout(pad=0) # Remove padding

    # Use the provided temporary directory
    spectrogram_filename = os.path.join(temp_dir, 'spectrogram.png')
    plt.savefig(spectrogram_filename, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()
    return spectrogram_filename


# Function to preprocess input for feature-based models
def preprocess_features(audio_data, sr):
    features = extract_features_segment(audio_data, sr)
    # The scaler expects a 2D array, even for a single sample
    features_scaled = scaler.transform(np.array([features]))
    return features_scaled

# Function to preprocess input for spectrogram-based models
def preprocess_spectrogram(audio_data, sr):
    # Ensure this input_shape matches what your spectrogram models expect
    input_shape = (64, 64, 3) 
    with tempfile.TemporaryDirectory() as temp_dir:
        spectrogram_filename = create_spectrogram(audio_data, sr, temp_dir, input_shape)
        # Load and resize the image
        spectrogram = tf.keras.preprocessing.image.load_img(
            spectrogram_filename, 
            target_size=(input_shape[0], input_shape[1])
        )
        spectrogram = tf.keras.preprocessing.image.img_to_array(spectrogram)
        spectrogram /= 255.0  # Normalize pixel values to [0, 1]
        spectrogram = np.expand_dims(spectrogram, axis=0) # Add batch dimension
    return spectrogram

# Streamlit app
st.set_page_config(layout="wide") # Use wide layout for better display
st.title("Music Genre Classification")
st.markdown("Upload an audio file (MP3, WAV, FLAC) and click 'Classify' to predict its music genre.")

# Upload audio file
audio_file = st.file_uploader("Upload audio file", type=['mp3', 'wav', 'FLAC'])

if audio_file:
    st.audio(audio_file, format='audio/wav')

    # Classify button
    if st.button("Classify"):
        status_text = st.empty() # Placeholder for status messages
        progress_bar = st.progress(0) # Initialize progress bar
        estimated_time_text = st.empty() # Placeholder for estimated time

        status_text.write("Loading audio file... This might take a moment.")
        
        # Load the audio file
        audio_data, sr = librosa.load(audio_file)
        
        # Define segment duration in samples (3 seconds as per model training)
        segment_duration_seconds = 3 
        segment_duration_samples = segment_duration_seconds * sr
        
        # Calculate the number of segments
        num_segments = len(audio_data) // segment_duration_samples

        if num_segments == 0:
            st.warning(f"Audio file is too short ({len(audio_data)/sr:.2f} seconds). "
                       f"Please upload an audio file at least {segment_duration_seconds} seconds long "
                       "to enable proper segmentation and classification.")
            status_text.empty()
            progress_bar.empty()
            estimated_time_text.empty()
            st.stop()


        
        # List to store ensemble predictions for each segment
        ensemble_predictions = []
        
        start_time = time.time() # Record start time for estimation

        for segment_index in range(num_segments):
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # Update progress bar
            progress_percent = (segment_index + 1) / num_segments
            progress_bar.progress(progress_percent)

            # Estimate remaining time
            if segment_index > 0: # Avoid division by zero on first iteration
                avg_time_per_segment = elapsed_time / (segment_index + 1)
                remaining_segments = num_segments - (segment_index + 1)
                estimated_remaining_time = avg_time_per_segment * remaining_segments
                estimated_time_text.write(
                    f"Estimated time remaining: {estimated_remaining_time:.2f} seconds "
                    f"(Elapsed: {elapsed_time:.2f} seconds)"
                )
            else:
                estimated_time_text.write("Estimating time...")


            start_sample = segment_index * segment_duration_samples
            end_sample = (segment_index + 1) * segment_duration_samples
            segment_audio = audio_data[start_sample:end_sample]

            # Predictions for each individual model
            features_input = preprocess_features(segment_audio, sr)
            spectrogram_input = preprocess_spectrogram(segment_audio, sr)

            # Get raw predictions from models (using verbose=0 to suppress Keras output)
            feature_model1_prediction = feature_model1.predict(features_input, verbose=0)
            feature_model2_prediction = feature_model2.predict(features_input, verbose=0)
            spectrogram_model1_prediction = spectrogram_model1.predict(spectrogram_input, verbose=0)
            spectrogram_model2_prediction = spectrogram_model2.predict(spectrogram_input, verbose=0)
            spectrogram_model3_prediction = spectrogram_model3.predict(spectrogram_input, verbose=0)
                
            # Average raw predictions for ensemble
            ensemble_prediction = (
                feature_model1_prediction +
                feature_model2_prediction +
                spectrogram_model1_prediction +
                spectrogram_model2_prediction +
                spectrogram_model3_prediction
            ) / 5
            ensemble_label = label_encoder.inverse_transform([np.argmax(ensemble_prediction)])[0]
            ensemble_predictions.append(ensemble_label)
        
        # After loop, clear status messages
        status_text.empty()
        estimated_time_text.empty()
        progress_bar.progress(1.0) # Ensure it reaches 100%

        # Find the most common ensemble prediction across all segments
        most_common_ensemble = Counter(ensemble_predictions).most_common(1)[0][0]


        st.markdown(f"**Classification Complete!**")
        st.info(f"## Predicted Genre is: **{most_common_ensemble}**")
