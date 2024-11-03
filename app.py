import streamlit as st
import PyPDF2
import requests
import json
import os
import io
from pydub import AudioSegment
from pydub.playback import play
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile

# Constants for Deepgram API
DEEPGRAM_API_KEY = "1ec8dd8fd6aebdb7a17f5e365e092f8b4e00414c"
DEEPGRAM_TTS_URL = "https://api.deepgram.com/v1/speak?model=aura-asteria-en"
DEEPGRAM_STT_URL = "https://api.deepgram.com/v1/listen?language=en&model=nova-2"

# Helper to fetch audio from Deepgram TTS API
def text_to_speech(text):
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "text/plain"
    }
    response = requests.post(DEEPGRAM_TTS_URL, headers=headers, data=text)
    
    # Convert to WAV format
    audio = AudioSegment.from_file(io.BytesIO(response.content), format="mp3")
    audio.export("question_audio.wav", format="wav")

# Helper to play audio
def play_audio(filename):
    audio = AudioSegment.from_wav(filename)
    play(audio)

# Function to record audio from the microphone
def record_audio(duration=15):
    fs = 44100  # Sample rate
    st.write("Recording answer...")
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')  # Change channels to 1 (mono)
    sd.wait()  # Wait until recording is finished
    # Save the audio data to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        wav.write(temp_audio.name, fs, audio_data)
        temp_audio_path = temp_audio.name
    return temp_audio_path

# Function to transcribe audio using Deepgram STT API
def speech_to_text(audio_path):
    with open(audio_path, "rb") as audio_file:
        headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "audio/wav"
        }
        response = requests.post(DEEPGRAM_STT_URL, headers=headers, data=audio_file)
    if response.status_code == 200:
        return response.json().get("results", {}).get("channels", [])[0].get("alternatives", [])[0].get("transcript", "")
    return "Transcription failed."

# Function to extract text from PDF
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# Function to send data to the local LLM and get questions
def generate_questions(resume_text, domain):
    url = "https://57e5-34-74-197-114.ngrok-free.app/api/generate/"
    prompt = (
        f"Generate 20 interview questions for a candidate's mock interview in the {domain} domain. "
        f"The questions should be based on the following resume text:\n\n{resume_text}\n\n"
        "Please provide 20 questions that align with the candidate's experience and the selected domain."
    )
    payload = {"model": "mistral", "prompt": prompt}
    response = requests.post(url, json=payload, stream=True)

    full_text = ""
    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode('utf-8'))
            full_text += data.get("response", "")
            if data.get("done", False):
                break

    questions = [q.strip() for q in full_text.splitlines() if q.strip()]
    return questions

# Function to get feedback from the LLM
def get_feedback(questions, responses):
    url = "https://57e5-34-74-197-114.ngrok-free.app/api/generate/"
    prompt = (
        "Here are the questions and the corresponding responses from the mock interview:\n\n"
    )
    for question, response in zip(questions, responses):
        prompt += f"Q: {question}\nA: {response}\n\n"
    prompt += "Please provide feedback and areas of improvement for the user."

    payload = {"model": "mistral", "prompt": prompt}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json().get("results", {}).get("channels", [])[0].get("alternatives", [])[0].get("transcript", "")
    return "Feedback request failed."

# Streamlit app layout
st.set_page_config(page_title="HR Mock Interview Generator", layout="wide")
st.title("HR Mock Interview Generator")
st.subheader("Upload your resume, select a domain, and get interview questions!")

# File uploader widget
uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")

if uploaded_file is not None:
    st.write(f"You uploaded: {uploaded_file.name}")
    resume_text = extract_text_from_pdf(uploaded_file)

    st.subheader("Select your interview domain")
    domain = st.selectbox("Choose your field:", ["IT", "EEE Core", "ECE Core", "CSE Core", "MECH Core"])

    if st.button("Generate Interview Questions"):
        questions = generate_questions(resume_text, domain)
        questions.insert(0, "Tell me about yourself.")  # Add the first question
        st.session_state["questions"] = questions
        st.session_state["current_question"] = 0
        st.session_state["responses"] = []
        st.session_state["first_question_asked"] = False  # Track if the first question has been asked

# Check if there are questions to ask
if "questions" in st.session_state and st.session_state["questions"]:
    current_question_index = st.session_state["current_question"]

    # Ask the first question immediately after generation if it hasn't been asked
    if not st.session_state["first_question_asked"]:
        question_text = st.session_state["questions"][0]
        st.subheader(f"Question 1: {question_text}")
        text_to_speech(question_text)
        st.write("Playing question audio...")
        play_audio("question_audio.wav")

        # Automatically record answer after the question audio
        answer_audio_path = record_audio(duration=15)
        user_response = speech_to_text(answer_audio_path)
        st.write(f"Your answer: {user_response}")
        st.session_state["responses"].append(user_response)

        st.session_state["first_question_asked"] = True  # Mark the first question as asked
        st.session_state["current_question"] += 1  # Move to the next question

    # Continue asking subsequent questions
    if st.session_state["first_question_asked"] and current_question_index < len(st.session_state["questions"]):
        question_text = st.session_state["questions"][current_question_index]

        # Display the current question
        st.subheader(f"Question {current_question_index + 1}")
        st.write(question_text)

        # Generate and play audio for the question
        text_to_speech(question_text)
        st.write("Playing question audio...")
        play_audio("question_audio.wav")

        # Automatically record answer after the question audio
        answer_audio_path = record_audio(duration=15)
        user_response = speech_to_text(answer_audio_path)
        st.write(f"Your answer: {user_response}")
        st.session_state["responses"].append(user_response)

        # Move to the next question if available
        if st.button("Next Question"):
            if current_question_index + 1 < len(st.session_state["questions"]):
                st.session_state["current_question"] += 1  # Move to the next question
            else:
                st.success("Interview complete! Here are your answers:")
                for idx, answer in enumerate(st.session_state["responses"], 1):
                    st.write(f"{idx}. {answer}")

                # Request feedback after completing the interview
                feedback = get_feedback(st.session_state["questions"], st.session_state["responses"])
                st.subheader("Feedback from the Interview:")
                st.write(feedback)  # Display the feedback
