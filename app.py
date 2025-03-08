import streamlit as st
import torch
import librosa
import gdown
import os
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from googleapiclient.discovery import build  # YouTube API client

# Function to download the model
def setup_model():
    file_id = "1Ibw2vhlORv_1XtcxiXGe3oJGrhr35Nvw"  # Google Drive file ID
    model_path = "my_wav2vec2_model"  # Path to save the model

    # Download only if the model does not exist
    if not os.path.exists(model_path):
        st.info("Downloading model... ⏳")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)

    return model_path

# Load the model
MODEL_PATH = setup_model()
processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_PATH)

# Emotion labels
EMOTION_LABELS = ["Fear", "Angry", "Disgust", "Neutral", "Sad", "Pleasant Surprise", "Happy"]

# Set up YouTube API
YOUTUBE_API_KEY = "AIzaSyAVEKC5VK9JiciYFylJ42eK5ONtojt0gjs"  # Replace with your API key
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# Function to fetch YouTube playlists
def get_youtube_playlists(emotion, max_results=3):
    search_query = f"{emotion} mood music playlist"
    request = youtube.search().list(q=search_query, part="snippet", type="playlist", maxResults=max_results)
    response = request.execute()

    playlists = []
    for item in response.get("items", []):
        title = item["snippet"]["title"]
        playlist_id = item["id"]["playlistId"]
        url = f"https://www.youtube.com/playlist?list={playlist_id}"
        playlists.append((title, url))
    
    return playlists

# Streamlit UI
st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎵", layout="centered")

st.markdown(
    """
    <style>
        .stApp { background-color: #021526; }
        h1, h2, h3 { color: #6EACDA; text-align: center; }
        div.stFileUploader { border: 2px solid #6EACDA; padding: 15px; border-radius: 10px; background-color: #03346E; }
        div.stFileUploader label { color: white; font-weight: bold; font-size: 16px; }
        div.stButton > button {
            background-color: #6EACDA; color: #021526; font-size: 18px; font-weight: bold;
            border-radius: 10px; padding: 12px; border: none; transition: 0.3s;
        }
        div.stButton > button:hover { background-color: #03346E; color: white; }
        div.stAudio { border: 2px solid #6EACDA; border-radius: 10px; padding: 10px; background-color: #03346E; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🎵 Speech Emotion Recognition & Music Recommendation")
st.write("Upload an audio file, and our AI will detect the emotion and suggest YouTube playlists to match your mood!")

# File uploader
uploaded_file = st.file_uploader("📂 Choose an audio file", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav", start_time=0)

    if st.button("🔍 Predict Emotion"):
        with st.spinner("Analyzing emotion... 🎭"):
            # Convert file to waveform
            audio, sr = librosa.load(uploaded_file, sr=16000)
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

            # Predict emotion
            with torch.no_grad():
                logits = model(**inputs).logits
                predicted_class = torch.argmax(logits, dim=1).item()

            # Display detected emotion
            detected_emotion = EMOTION_LABELS[predicted_class]
            st.success(f"🎭 Detected Emotion: **{detected_emotion.upper()}**")

            # Fetch and display recommended playlists
            st.subheader("🎶 Recommended Playlists")
            playlists = get_youtube_playlists(detected_emotion)
            if playlists:
                for title, url in playlists:
                    st.markdown(f"[🎵 {title}]({url})", unsafe_allow_html=True)
            else:
                st.warning("⚠️ No playlists found. Try again later.")
