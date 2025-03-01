import streamlit as st
import pickle
import numpy as np
import librosa
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH")
ENCODER_PATH = os.getenv("ENCODER_PATH")

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
    zcr = librosa.feature.zero_crossing_rate(y)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6, fmin=50)
    
    mfccs_mean = np.mean(mfccs.T, axis=0)
    chroma_mean = np.mean(chroma.T, axis=0)
    zcr_mean = np.mean(zcr.T, axis=0)
    spec_contrast_mean = np.mean(spec_contrast.T, axis=0)
    
    return np.concatenate((mfccs_mean, chroma_mean, zcr_mean, spec_contrast_mean))

clf = pickle.load(open(MODEL_PATH, 'rb'))
label_encoder = pickle.load(open(ENCODER_PATH, 'rb'))

def predict_audio(file_path):
    feature = extract_features(file_path)
    feature = feature.reshape(1, -1)
    prediction = clf.predict(feature)
    return label_encoder.inverse_transform(prediction)[0]

st.title("Baby Cry Reason Predictor👶🏻")
st.write("Upload an audio file to predict the reason behind the baby's cry.")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    if st.button("Predict"):  # Button to trigger prediction
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
        
        prediction = predict_audio(temp_file_path)
        st.success(f'The reason behind the baby cry is: {prediction}')
        st.balloons()
        os.remove(temp_file_path)