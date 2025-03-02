import streamlit as st
import pickle
import numpy as np
import tempfile
import os
from pydub import AudioSegment
from scipy.io import wavfile
from scipy.signal import find_peaks
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv

load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH")
ENCODER_PATH = os.getenv("ENCODER_PATH")

def extract_features(file_path):
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_channels(1).set_frame_rate(22050)  # Ensure mono & set sample rate
    audio.export("temp.wav", format="wav")

    sample_rate, samples = wavfile.read("temp.wav")

    samples = samples.astype(np.float32)

    mfcc_like = dct(samples, type=2, norm='ortho')[1:14]

    fft_spectrum = np.abs(np.fft.fft(samples))[:len(samples) // 2]
    chroma_like = np.mean(fft_spectrum.reshape(-1, 12), axis=0) 

    zero_crossings = np.mean(np.abs(np.diff(np.sign(samples))))

    spectral_contrast = np.std(fft_spectrum.reshape(-1, 6), axis=0)  

    return np.concatenate((mfcc_like, chroma_like, [zero_crossings], spectral_contrast))

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
