import streamlit as st
import PyPDF2
import requests
import tempfile
import base64
from st_audiorec import st_audiorec
from faster_whisper import WhisperModel
from groq import Groq
import re
import time

st.set_page_config(page_title="🎙 Voice Bot", layout="centered")

st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap" rel="stylesheet">
    <style>
    body {
        background: linear-gradient(135deg, #ff8a00, #e52e71);
        font-family: 'Press Start 2P', cursive;
        color: #fff;
    }
    .stApp {
        background-color: #1e1e1e;
    }
    .css-18ni7ap.e8zbici2 {
        background-color: #2d2d2d;
        border-radius: 15px;
        padding: 15px;
        box-shadow: 0px 0px 15px #ff00ff;
    }
    .stButton>button {
        color: #fff;
        background-color: #ff00ff;
        border-radius: 25px;
        padding: 10px 20px;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff66cc;
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)

HUME_API_KEY = st.secrets.get("HUME_API_KEY", "")
GROQ_API_KEY = st.secrets["GROQ_KEY"]
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.1-8b-instant"

@st.cache_resource
def load_whisper_model():
    return WhisperModel("tiny", device="cpu", compute_type="int8")

whisper_model = load_whisper_model()

@st.cache_resource
def load_groq_client():
    return Groq(api_key=GROQ_API_KEY)

groq_client = load_groq_client()

st.title("🎙 Personalized Voice Bot (Groq LLM + TTS + Whisper)")

def extract_pdf_text(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()

def transcribe_audio_faster_whisper(audio_bytes):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        tmpfile.write(audio_bytes)
        tmpfile.flush()
        segments, _ = whisper_model.transcribe(tmpfile.name)
        transcription = " ".join(segment.text for segment in segments)
    return transcription

def generate_response_groq_direct(prompt):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150,
        "temperature": 0.5
    }

    try:
        response = requests.post(GROQ_CHAT_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        st.error("⚠ Error generating response from Groq.")
        st.stop()

def parse_groq_wait_time(wait_str):
    pattern = r'(?:(\d+)h)?(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?'
    match = re.match(pattern, wait_str)
    if not match:
        return 0
    hours = int(match.group(1)) if match.group(1) else 0
    minutes = int(match.group(2)) if match.group(2) else 0
    seconds = float(match.group(3)) if match.group(3) else 0.0
    return hours * 3600 + minutes * 60 + seconds

def synthesize_tts_file(text, voice="Fritz-PlayAI", fmt="wav"):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "playai-tts",
        "input": text,
        "voice": voice,
        "response_format": fmt
    }

    try:
        resp = requests.post("https://api.groq.com/openai/v1/audio/speech", headers=headers, json=payload)
        if resp.status_code == 429:
            msg = resp.json()['error'].get('message', '')
            wait_str_match = re.search(r'in ([\dhms\.]+)', msg)
            if wait_str_match:
                wait_str = wait_str_match.group(1)
                wait_secs = parse_groq_wait_time(wait_str)
                st.warning(f"Rate limit hit. Waiting for {int(wait_secs)} seconds before retrying...")
                time.sleep(wait_secs)
                return synthesize_tts_file(text, voice, fmt)
            else:
                st.error(f"Rate limit hit, but couldn't parse wait time: {msg}")
                st.stop()
        resp.raise_for_status()
        return resp.content
    except requests.exceptions.HTTPError as err:
        st.error(f"TTS Error: {resp.status_code} - {resp.text}")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected TTS error: {e}")
        st.stop()

def autoplay_audio_bytes(audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        tmpfile.write(audio_bytes)
        tmpfile_path = tmpfile.name

    with open(tmpfile_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true">
              <source src="data:audio/wav;base64,{b64}" type="audio/wav">
              Your browser does not support the audio element.
            </audio>
        """
        st.markdown(md, unsafe_allow_html=True)

    import os
    os.unlink(tmpfile_path)

with st.sidebar:
    st.header("📄 Upload Document (PDF)")
    pdf_file = st.file_uploader("Choose a PDF", type="pdf")
    resume_text = ""
    if pdf_file:
        resume_text = extract_pdf_text(pdf_file)
        st.success("✅ Document uploaded and extracted!")

    st.markdown("### 🎙 Choose TTS Voice")
    voice = st.selectbox("Voice", [
        "Fritz-PlayAI", "Celeste-PlayAI", "Atlas-PlayAI", "Briggs-PlayAI"
    ], index=0)

st.subheader("🎤 Record Your Question")
audio_bytes = st_audiorec()

if audio_bytes is not None:
    if not resume_text:
        st.warning("Please upload your document PDF first.")
        st.stop()

    with st.spinner("🔍 Transcribing audio locally..."):
        transcription = transcribe_audio_faster_whisper(audio_bytes)
    st.success(f"📝 Transcription: {transcription}")

    prompt = f"""
    You are a helpful voice assistant that answers spoken questions either based on a document (like a resume) or from general knowledge.

    Instructions:
    - If the question is about the uploaded document, answer as if *you are the person described* — use natural first-person tone, but *do not repeat the person's name*.
    - If the question is unrelated to the document, give a short, clear factual answer.

    Keep the response concise (under 150 words) and natural.

    Document:
    {resume_text}

    Question:
    {transcription}
    """

    with st.spinner("💡 Generating response with Groq..."):
        reply = generate_response_groq_direct(prompt)

    st.success("✅ Response generated")
    st.markdown("### 🗣 Response")
    st.markdown(f"<div style='font-size:18px; padding:10px;'>{reply}</div>", unsafe_allow_html=True)

    with st.spinner("🔈 Synthesizing speech with Groq TTS..."):
        audio_response = synthesize_tts_file(reply, voice=voice, fmt="wav")

    autoplay_audio_bytes(audio_response)

st.caption("⚡ Powered by Groq (LLM + TTS), faster-whisper (STT)")