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


# Load saved models
feature_model1 = load_model('ann_feature_10sec_model.keras')
feature_model2 = load_model('cnn_feature_10sec_model.keras')
spectrogram_model1 = load_model('CNN_spectrogram_10sec_model.keras')
spectrogram_model2 = load_model('GRU_spectrogram_10sec_model.keras')
spectrogram_model3 = load_model('LSTM_spectrogram_10sec_model.keras')


# Load scaler and label encoder for feature-based models
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

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

    # Ensure tempo is stored as a float
    tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sr)
    tempo = float(tempo)  # Fix: Convert tempo to float

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

    # Use the provided temporary directory
    spectrogram_filename = os.path.join(temp_dir, 'spectrogram.png')
    plt.savefig(spectrogram_filename, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

    return spectrogram_filename


# Function to preprocess input for feature-based models
def preprocess_features(audio_data, sr):
    features = extract_features_segment(audio_data, sr)

    features_scaled = scaler.transform(np.array([features]))
    return features_scaled

# Function to preprocess input for spectrogram-based models
def preprocess_spectrogram(audio_data, sr):
    input_shape = (64, 64, 3)  # Assuming input shape for spectrogram models
    with tempfile.TemporaryDirectory() as temp_dir:
        spectrogram_filename = create_spectrogram(audio_data, sr, temp_dir, input_shape)
        spectrogram = tf.keras.preprocessing.image.load_img(spectrogram_filename, target_size=(input_shape[0], input_shape[1]))
        spectrogram = tf.keras.preprocessing.image.img_to_array(spectrogram)
        spectrogram /= 255.0  # Normalize pixel values to [0, 1]
        spectrogram = np.expand_dims(spectrogram, axis=0)
    return spectrogram

# Streamlit app
st.title("Music Genre Classification")

# Upload audio file
audio_file = st.file_uploader("Upload audio file", type=['mp3', 'wav', 'FLAC'])

if audio_file:
    st.audio(audio_file, format='audio/wav')

    # Classify button
    if st.button("Classify"):
        st.write("Classifying... Please wait")

        # Load the audio file
        audio_data, sr = librosa.load(audio_file)

        # Define segment duration (10 seconds)
        segment_duration = 3 * sr

        # Calculate the number of segments
        num_segments = len(audio_data) // segment_duration

        # Define lists to store predictions for each model
        feature_model1_predictions = []
        feature_model2_predictions = []
        spectrogram_model1_predictions = []
        spectrogram_model2_predictions = []
        spectrogram_model3_predictions = []
        ensemble_predictions = []
    
        for segment_index in range(num_segments):
            start_sample = segment_index * segment_duration
            end_sample = (segment_index + 1) * segment_duration
            segment_audio = audio_data[start_sample:end_sample]
    
            # Predictions for each individual model
            features_input = preprocess_features(segment_audio, sr)
            spectrogram_input = preprocess_spectrogram(segment_audio, sr)
    
            # Get raw predictions from models
            feature_model1_prediction = feature_model1.predict(features_input)
            feature_model2_prediction = feature_model2.predict(features_input)
            spectrogram_model1_prediction = spectrogram_model1.predict(spectrogram_input)
            spectrogram_model2_prediction = spectrogram_model2.predict(spectrogram_input)
            spectrogram_model3_prediction = spectrogram_model3.predict(spectrogram_input)
    
            # Convert raw predictions to labels
            feature_model1_label = label_encoder.inverse_transform([np.argmax(feature_model1_prediction)])[0]
            feature_model2_label = label_encoder.inverse_transform([np.argmax(feature_model2_prediction)])[0]
            spectrogram_model1_label = label_encoder.inverse_transform([np.argmax(spectrogram_model1_prediction)])[0]
            spectrogram_model2_label = label_encoder.inverse_transform([np.argmax(spectrogram_model2_prediction)])[0]
            spectrogram_model3_label = label_encoder.inverse_transform([np.argmax(spectrogram_model3_prediction)])[0]
    
            # Append label predictions to respective lists
            feature_model1_predictions.append(feature_model1_label)
            feature_model2_predictions.append(feature_model2_label)
            spectrogram_model1_predictions.append(spectrogram_model1_label)
            spectrogram_model2_predictions.append(spectrogram_model2_label)
            spectrogram_model3_predictions.append(spectrogram_model3_label)
    
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

        # Find the most common ensemble prediction
        most_common_ensemble = Counter(ensemble_predictions).most_common(1)[0][0]
        st.success("Most Common ensemble_model Prediction: {}".format(most_common_ensemble))
