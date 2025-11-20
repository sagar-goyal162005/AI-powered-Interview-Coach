import streamlit as st

# Handle audio module imports with fallbacks
try:
    import speech_recognition as sr
except ImportError:
    sr = None
    
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import numpy as np
try:
    import sounddevice as sd  # Requires PortAudio; may not be available on hosted platforms
    _SOUNDDEVICE_AVAILABLE = True
except Exception:
    _SOUNDDEVICE_AVAILABLE = False

try:
    import wavio
except ImportError:
    wavio = None
    
import re
import threading
import queue
import time
import os
import hashlib
import json
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import requests
from PIL import Image
import io
import tempfile
import moviepy.editor as mp
from gtts import gTTS
from pydub import AudioSegment
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass
try:
    from openai import OpenAI
    _OPENAI_NEW = True
except ImportError:
    import openai
    _OPENAI_NEW = False

# Ensure necessary NLTK resources are available (auto-download if missing)
def ensure_nltk_resources():
    resources = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),  # newer NLTK versions separate tables
        ('corpora/stopwords', 'stopwords'),
        ('sentiment/vader_lexicon', 'vader_lexicon'),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)

ensure_nltk_resources()

# API keys for mock interview generation
openai_api_key = os.getenv("OPENAI_API_KEY")
freepik_api_key = os.getenv("FREEPIK_API_KEY")

# Initialize OpenAI client
client = None
if _OPENAI_NEW and openai_api_key:
    try:
        client = OpenAI(api_key=openai_api_key, base_url="https://api.aimlapi.com/v1")
    except Exception as e:
        pass
elif openai_api_key:
    openai.api_key = openai_api_key
    openai.api_base = "https://api.aimlapi.com/v1"

# Configuration for WebRTC
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Database setup (using JSON file for simplicity)
USER_DB_FILE = "users.json"
REPORTS_DIR = "reports"

def init_db():
    if not os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, "w") as f:
            json.dump({}, f)
    
    # Create reports directory if it doesn't exist
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)

def hash_password(password):
    """Create a SHA-256 hash of the password"""
    return hashlib.sha256(password.encode()).hexdigest()

def validate_password(password):
    """Validate password meets requirements"""
    # At least 8 characters, one uppercase, one number, one special character
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter"
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one number"
    if not any(c in "!@#$%^&*()-_=+[]{}|;:,.<>?/" for c in password):
        return False, "Password must contain at least one special character"
    return True, "Password is valid"

def validate_username(username):
    """Validate username meets requirements"""
    if len(username) < 8:
        return False, "Username must be at least 8 characters"
    return True, "Username is valid"

def user_exists(username):
    """Check if a user already exists"""
    with open(USER_DB_FILE, "r") as f:
        users = json.load(f)
    return username in users

def add_user(username, password):
    """Add a new user to the database"""
    with open(USER_DB_FILE, "r") as f:
        users = json.load(f)
    
    users[username] = hash_password(password)
    
    with open(USER_DB_FILE, "w") as f:
        json.dump(users, f)

def authenticate_user(username, password):
    """Authenticate a user"""
    with open(USER_DB_FILE, "r") as f:
        users = json.load(f)
    
    if username in users and users[username] == hash_password(password):
        return True
    return False

def get_user_reports_path(username):
    """Get the path to a user's reports file"""
    return os.path.join(REPORTS_DIR, f"{username}_reports.json")

def save_interview_report(username, report_data):
    """Save an interview report for a user"""
    reports_file = get_user_reports_path(username)
    
    # Load existing reports or create new list
    if os.path.exists(reports_file):
        with open(reports_file, "r") as f:
            reports = json.load(f)
    else:
        reports = []
    
    # Add timestamp to report data
    report_data["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    report_data["date"] = time.strftime("%Y-%m-%d")
    
    # Add the new report
    reports.append(report_data)
    
    # Save updated reports
    with open(reports_file, "w") as f:
        json.dump(reports, f, indent=4)
    
    return True

def get_user_reports(username):
    """Get all reports for a user"""
    reports_file = get_user_reports_path(username)
    
    if os.path.exists(reports_file):
        with open(reports_file, "r") as f:
            return json.load(f)
    else:
        return []

# Enhanced Video frame transformer for webcam with posture analysis
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.posture_status = "Unknown"
        self.shoulder_slope = 0
        self.head_position = "Unknown"
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.last_posture_check = time.time()
        self.check_interval = 1.0  # Check posture every second
        self.posture_history = []  # Store recent posture assessments
        self.slouch_counter = 0
        self.posture_feedback = "Analyzing your posture..."
        self.last_face_seen = time.time()
        # Pre-create CLAHE for light enhancement
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    def detect_face_and_shoulders(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None, None
        
        # Use the largest face detected
        face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = face
        
        # Approximate shoulder positions based on face position
        face_center_x = x + w//2
        face_bottom_y = y + h
        
        # Estimate shoulder area (wider than face and below it)
        shoulder_y = face_bottom_y + h//2
        shoulder_width = w * 2.5
        left_shoulder_x = max(0, int(face_center_x - shoulder_width//2))
        right_shoulder_x = min(img.shape[1], int(face_center_x + shoulder_width//2))
        
        return (x, y, w, h), (left_shoulder_x, right_shoulder_x, shoulder_y)
    
    def analyze_posture(self, img):
        face_rect, shoulder_points = self.detect_face_and_shoulders(img)
        
        if face_rect is None or shoulder_points is None:
            # If we recently saw a face keep previous status else notify user
            if time.time() - self.last_face_seen > 3:
                self.posture_feedback = "Face not detected – sit facing camera with good lighting."
                self.posture_status = "Not Visible"
            return self.posture_status, 0, "Unknown", img
        
        x, y, w, h = face_rect
        left_shoulder_x, right_shoulder_x, shoulder_y = shoulder_points
        
        # Draw face rectangle
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Draw approximate shoulder line
        cv2.line(img, (left_shoulder_x, shoulder_y), (right_shoulder_x, shoulder_y), (0, 255, 0), 2)
        
        # Calculate slope of shoulder line (ideally should be close to 0 for good posture)
        # For now, we're using a horizontal line, so slope is 0
        shoulder_slope = 0
        
        # Analyze head position relative to shoulders
        face_center_x = x + w//2
        shoulders_center_x = (left_shoulder_x + right_shoulder_x) // 2
        
        # Calculate horizontal offset between face center and shoulders center
        horizontal_offset = abs(face_center_x - shoulders_center_x)
        
        # Determine head position
        if horizontal_offset < w * 0.2:  # If offset is small
            head_position = "Centered"
        else:
            head_position = "Tilted" if face_center_x < shoulders_center_x else "Leaning"
        
        # Determine overall posture
        if head_position == "Centered" and abs(shoulder_slope) < 0.1:
            posture_status = "Good"
            self.slouch_counter = max(0, self.slouch_counter - 1)
        else:
            posture_status = "Needs Improvement"
            self.slouch_counter += 1

        self.last_face_seen = time.time()
        
        # Add posture assessment to history
        self.posture_history.append(posture_status)
        if len(self.posture_history) > 10:  # Keep only last 10 assessments
            self.posture_history.pop(0)
        
        # Generate specific feedback
        if posture_status == "Good":
            self.posture_feedback = "Great posture! Keep it up!"
        elif head_position != "Centered":
            self.posture_feedback = "Try to keep your head centered above your shoulders."
        else:
            self.posture_feedback = "Straighten your shoulders for better posture."
        
        # Count how many recent posture checks were "Needs Improvement"
        if self.slouch_counter > 5:
            self.posture_feedback = "You're slouching! Sit up straight for a professional appearance."
        
        return posture_status, shoulder_slope, head_position, img
    
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Improve image quality (contrast enhancement)
        try:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            cl = self.clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        except Exception:
            pass  # Fail silently if enhancement fails
        
        # Only analyze posture every check_interval seconds to reduce processing load
        current_time = time.time()
        if current_time - self.last_posture_check > self.check_interval:
            self.posture_status, self.shoulder_slope, self.head_position, img = self.analyze_posture(img)
            self.last_posture_check = current_time
        
        # Add interview tips and posture feedback at the top of the video
        h, w = img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Create a semi-transparent overlay for text background
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        alpha = 0.7  # Transparency factor
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        
        # Add interview tip
        tip_text = "Maintain eye contact and show confidence!"
        cv2.putText(img, tip_text, (10, 25), font, 0.6, (0, 255, 0), 2)
        
        # Add posture feedback with color based on status
        if self.posture_status == "Good":
            color = (0, 255, 0)  # Green for good posture
        else:
            color = (0, 165, 255)  # Orange for needs improvement
            
        cv2.putText(img, f"Posture: {self.posture_status} - {self.posture_feedback}", 
                   (10, 60), font, 0.6, color, 2)
        
        return img

    def posture_recommendations(self):
        """Generate posture improvement recommendations based on recent history."""
        history = self.posture_history[-10:]
        if not history:
            return ["Ensure your full face is visible to begin analysis."]
        improvements = []
        needs_impr_ratio = history.count("Needs Improvement") / len(history)
        if needs_impr_ratio > 0.6:
            improvements.append("Keep shoulders level – imagine a string pulling the top of your head up.")
        if self.head_position in ("Tilted", "Leaning"):
            improvements.append("Center your head above your shoulders to appear attentive.")
        if self.slouch_counter > 3:
            improvements.append("Roll shoulders back and sit upright to reduce slouching.")
        if self.posture_status == "Good" and not improvements:
            improvements.append("Maintain current posture – great alignment!")
        return improvements

class InterviewCoach:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.stopwords = set(stopwords.words('english'))

        self.professional_words = {
            'accomplished', 'achieved', 'analyzed', 'coordinated', 'created',
            'delivered', 'developed', 'enhanced', 'executed', 'improved',
            'initiated', 'launched', 'managed', 'optimized', 'organized',
            'planned', 'resolved', 'spearheaded', 'streamlined', 'success'
        }

        self.filler_words = {
            'um', 'uh', 'like', 'you know', 'actually', 'basically', 'literally',
            'sort of', 'kind of', 'so', 'well', 'just', 'stuff', 'things'
        }

        self.sample_rate = 44100  # Hz
        self.duration = 30  # seconds
        self.recording_in_progress = False
        self.recording_thread = None
        self.stop_recording = False

    def record_voice(self, filename="interview_response.wav"):
        """Record audio locally if sounddevice/PortAudio is available.
        On hosted platforms (Render, Streamlit Cloud) this will gracefully disable recording.
        """
        if not _SOUNDDEVICE_AVAILABLE:
            print("Sound recording unavailable (PortAudio not installed). Skipping microphone capture.")
            return None
        self.recording_in_progress = True
        self.stop_recording = False
        print(f"Recording your answer for {self.duration} seconds...")
        print("Speak now!")
        total_samples = int(self.duration * self.sample_rate)
        recording = np.zeros((total_samples, 1))
        chunk_size = int(0.1 * self.sample_rate)
        recorded_samples = 0
        try:
            stream = sd.InputStream(samplerate=self.sample_rate, channels=1)
            stream.start()
            while recorded_samples < total_samples and not self.stop_recording:
                samples_to_record = min(chunk_size, total_samples - recorded_samples)
                data, overflowed = stream.read(samples_to_record)
                recording[recorded_samples:recorded_samples + len(data)] = data
                recorded_samples += len(data)
            stream.stop()
            stream.close()
        except Exception as e:
            print(f"Audio recording failed: {e}")
            return None
        if recorded_samples > self.sample_rate * 0.5:
            recording = recording[:recorded_samples]
            wavio.write(filename, recording, self.sample_rate, sampwidth=2)
            print(f"Recording saved to {filename}")
            return filename
        print("Recording canceled or too short")
        return None

    def transcribe_audio(self, audio_file):
        print("Transcribing audio...")
        if audio_file is None:
            return ""
            
        with sr.AudioFile(audio_file) as source:
            audio_data = self.recognizer.record(source)
            try:
                text = self.recognizer.recognize_google(audio_data)
                print("Transcription complete!")
                return text
            except sr.UnknownValueError:
                print("Speech Recognition could not understand audio")
                return ""
            except sr.RequestError as e:
                print(f"Could not request results from Speech Recognition service; {e}")
                return ""

    def analyze_tone(self, text):
        if not text:
            return {
                'score': 0,
                'sentiment': 'neutral',
                'feedback': 'No speech detected to analyze tone.'
            }

        sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
        compound_score = sentiment_scores['compound']

        if compound_score >= 0.05:
            sentiment = 'positive'
        elif compound_score <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        feedback = ""
        if sentiment == 'positive':
            feedback = "Your tone is positive and enthusiastic, which is great for an interview. Keep up the energy!"
            if compound_score > 0.5:
                feedback += " However, be careful not to come across as overly enthusiastic as it might seem insincere."
        elif sentiment == 'negative':
            feedback = "Your tone comes across as somewhat negative. Try to use more positive language and emphasize your strengths and achievements."
        else:
            feedback = "Your tone is neutral. While this is professional, try to inject some enthusiasm when discussing your achievements or interest in the role."

        return {
            'score': compound_score,
            'sentiment': sentiment,
            'feedback': feedback
        }

    def analyze_word_choice(self, text):
        if not text:
            return {
                'professional_word_count': 0,
                'filler_word_count': 0,
                'professional_words_used': [],
                'filler_words_used': [],
                'feedback': 'No speech detected to analyze word choice.'
            }

        words = nltk.word_tokenize(text.lower())
        professional_words_used = [word for word in words if word in self.professional_words]
        filler_words_used = [filler for filler in self.filler_words if filler in text.lower()]

        feedback = ""
        if professional_words_used:
            feedback += f"Good use of professional language! Words like {', '.join(professional_words_used[:3])} strengthen your responses. "
        else:
            feedback += "Consider incorporating more professional language to highlight your skills and achievements. "

        if filler_words_used:
            feedback += f"Try to reduce filler words/phrases like {', '.join(filler_words_used[:3])}. These can make you sound less confident."
        else:
            feedback += "You've done well avoiding filler words, which makes your speech sound more confident and prepared."

        return {
            'professional_word_count': len(professional_words_used),
            'filler_word_count': len(filler_words_used),
            'professional_words_used': professional_words_used,
            'filler_words_used': filler_words_used,
            'feedback': feedback
        }

    def analyze_confidence(self, text, tone_analysis):
        if not text:
            return {
                'confidence_score': 0,
                'feedback': 'No speech detected to analyze confidence.'
            }

        confidence_score = 5  # Base score out of 10
        sentiment_score = tone_analysis['score']
        if sentiment_score > 0:
            confidence_score += sentiment_score * 2
        elif sentiment_score < -0.2:
            confidence_score -= abs(sentiment_score) * 2

        hesitation_patterns = [
            r'\bI think\b', r'\bmaybe\b', r'\bpossibly\b', r'\bperhaps\b',
            r'\bI guess\b', r'\bsort of\b', r'\bkind of\b', r'\bI hope\b',
            r'\bI\'m not sure\b', r'\bI don\'t know\b'
        ]

        hesitation_count = sum(len(re.findall(pattern, text.lower())) for pattern in hesitation_patterns)
        confidence_score -= hesitation_count * 0.5

        sentences = nltk.sent_tokenize(text)
        avg_sentence_length = np.mean([len(nltk.word_tokenize(sentence)) for sentence in sentences]) if sentences else 0

        if avg_sentence_length > 20:
            confidence_score += 1
        elif avg_sentence_length < 8:
            confidence_score -= 1

        confidence_score = max(0, min(10, confidence_score))

        if confidence_score >= 8:
            feedback = "You sound very confident. Your delivery is strong and assertive."
        elif confidence_score >= 6:
            feedback = "You sound reasonably confident. With a few adjustments, you could project even more authority."
        elif confidence_score >= 4:
            feedback = "Your confidence level seems moderate. Try speaking more assertively and avoiding hesitant language."
        else:
            feedback = "You may want to work on projecting more confidence. Try reducing hesitant phrases and speaking with more conviction."

        return {
            'confidence_score': confidence_score,
            'feedback': feedback
        }

    def provide_comprehensive_feedback(self, analysis_results):
        tone = analysis_results['tone']
        word_choice = analysis_results['word_choice']
        confidence = analysis_results['confidence']
        posture = analysis_results.get('posture', {'status': 'Not analyzed', 'feedback': 'No posture analysis available.'})

        feedback_text = "\n" + "=" * 50 + "\n"
        feedback_text += "INTERVIEW RESPONSE EVALUATION\n"
        feedback_text += "=" * 50 + "\n\n"

        feedback_text += "TONE ANALYSIS:\n"
        feedback_text += f"Sentiment: {tone['sentiment']} (Score: {tone['score']:.2f})\n"
        feedback_text += f"Feedback: {tone['feedback']}\n\n"

        feedback_text += "WORD CHOICE ANALYSIS:\n"
        feedback_text += f"Professional words used: {word_choice['professional_word_count']}\n"
        if word_choice['professional_words_used']:
            feedback_text += f"Examples: {', '.join(word_choice['professional_words_used'][:3])}\n"

        feedback_text += f"Filler words/phrases used: {word_choice['filler_word_count']}\n"
        if word_choice['filler_words_used']:
            feedback_text += f"Examples: {', '.join(word_choice['filler_words_used'][:3])}\n"

        feedback_text += f"Feedback: {word_choice['feedback']}\n\n"

        feedback_text += "CONFIDENCE ASSESSMENT:\n"
        feedback_text += f"Confidence Score: {confidence['confidence_score']:.1f}/10\n"
        feedback_text += f"Feedback: {confidence['feedback']}\n\n"

        # Include posture feedback if available
        if 'posture' in analysis_results:
            feedback_text += "POSTURE ASSESSMENT:\n"
            feedback_text += f"Status: {posture['status']}\n"
            feedback_text += f"Feedback: {posture['feedback']}\n\n"

        avg_score = (tone['score'] + 1) * 5 + confidence['confidence_score']
        avg_score /= 2

        if avg_score >= 8:
            feedback_text += "Excellent interview response! You presented yourself very well.\n"
        elif avg_score >= 6:
            feedback_text += "Good interview response. With some minor improvements, you'll make an even stronger impression.\n"
        elif avg_score >= 4:
            feedback_text += "Acceptable interview response. Focus on the improvement areas mentioned above.\n"
        else:
            feedback_text += "Your interview response needs improvement. Consider practicing more with the suggestions provided.\n"

        feedback_text += "\nAREAS TO FOCUS ON:\n"
        improvement_areas = []

        if tone['score'] < 0:
            improvement_areas.append("Using more positive language")
        if word_choice['filler_word_count'] > 3:
            improvement_areas.append("Reducing filler words/phrases")
        if word_choice['professional_word_count'] < 2:
            improvement_areas.append("Incorporating more professional vocabulary")
        if confidence['confidence_score'] < 5:
            improvement_areas.append("Building confidence in delivery")
        if 'posture' in analysis_results and posture['status'] != 'Good':
            improvement_areas.append("Improving your posture and body language")

        if improvement_areas:
            for i, area in enumerate(improvement_areas, 1):
                feedback_text += f"{i}. {area}\n"
        else:
            feedback_text += "Great job! Keep practicing to maintain your strong performance.\n"

        feedback_text += "=" * 50 + "\n"

        return feedback_text

    def analyze_text_input(self, text):
        tone_analysis = self.analyze_tone(text)
        word_choice_analysis = self.analyze_word_choice(text)
        confidence_analysis = self.analyze_confidence(text, tone_analysis)

        analysis_results = {
            'tone': tone_analysis,
            'word_choice': word_choice_analysis,
            'confidence': confidence_analysis,
            'text': text
        }

        return analysis_results

def create_login_page():
    st.subheader("Login")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    
    if st.button("Login", key="login_button"):
        if username and password:
            if authenticate_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"Welcome back, {username}!")
                st.rerun()
            else:
                st.error("Invalid username or password")
        else:
            st.warning("Please enter both username and password")

def create_signup_page():
    st.subheader("Create an Account")
    
    username = st.text_input("Username (minimum 8 characters)", key="signup_username")
    password = st.text_input("Password", type="password", help="Password must contain at least 8 characters, one uppercase letter, one number, and one special character", key="signup_password")
    confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
    
    if st.button("Sign Up", key="Sign_up_button"):
        if not username or not password or not confirm_password:
            st.warning("Please fill in all fields")
            return
            
        valid_username, username_msg = validate_username(username)
        if not valid_username:
            st.error(username_msg)
            return
            
        valid_password, password_msg = validate_password(password)
        if not valid_password:
            st.error(password_msg)
            return
            
        if password != confirm_password:
            st.error("Passwords do not match")
            return
            
        if user_exists(username):
            st.error("Username already exists. Please choose another one.")
            return
            
        add_user(username, password)
        st.success("Account created successfully! Please log in.")
        st.session_state.show_login = True
        st.rerun()

def create_progress_report_pdf(username, reports, start_date=None, end_date=None):
    """Generate a PDF report of user progress"""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from io import BytesIO
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    
    # Add title
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    elements.append(Paragraph(f"Interview Performance Report for {username}", title_style))
    elements.append(Spacer(1, 20))
    
    # Filter reports by date if specified
    if start_date and end_date:
        filtered_reports = [r for r in reports if start_date <= r["date"] <= end_date]
    else:
        filtered_reports = reports
    
    if not filtered_reports:
        elements.append(Paragraph("No data available for the selected date range.", styles['Normal']))
    else:
        # Add summary information
        num_sessions = len(filtered_reports)
        avg_confidence = sum(r["analysis"]["confidence"]["confidence_score"] for r in filtered_reports) / num_sessions if num_sessions > 0 else 0
        avg_sentiment = sum(r["analysis"]["tone"]["score"] for r in filtered_reports) / num_sessions if num_sessions > 0 else 0
        
        elements.append(Paragraph(f"Sessions Analyzed: {num_sessions}", styles['Normal']))
        elements.append(Paragraph(f"Average Confidence Score: {avg_confidence:.1f}/10", styles['Normal']))
        elements.append(Paragraph(f"Average Tone Score: {avg_sentiment:.2f} (-1 to 1 scale)", styles['Normal']))
        elements.append(Spacer(1, 20))
        
        # Add performance over time table
        elements.append(Paragraph("Performance Over Time", styles['Heading2']))
        elements.append(Spacer(1, 10))
        
        # Create table data
        table_data = [
            ["Date", "Question", "Confidence", "Tone", "Professional Words", "Filler Words"]
        ]
        
        for report in filtered_reports:
            date = report["date"]
            question = report["question"][:30] + "..." if len(report["question"]) > 30 else report["question"]
            confidence = f"{report['analysis']['confidence']['confidence_score']:.1f}"
            tone = f"{report['analysis']['tone']['score']:.2f}"
            prof_words = str(report['analysis']['word_choice']['professional_word_count'])
            filler_words = str(report['analysis']['word_choice']['filler_word_count'])
            
            table_data.append([date, question, confidence, tone, prof_words, filler_words])
        
        # Create table
        table = Table(table_data, colWidths=[80, 150, 80, 80, 80, 80])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 20))
        
        # Add improvement areas
        elements.append(Paragraph("Common Areas for Improvement", styles['Heading2']))
        elements.append(Spacer(1, 10))
        
        # Count occurrence of each improvement area
        improvement_areas = {}
        for report in filtered_reports:
            tone_score = report["analysis"]["tone"]["score"]
            filler_count = report["analysis"]["word_choice"]["filler_word_count"]
            prof_word_count = report["analysis"]["word_choice"]["professional_word_count"]
            confidence_score = report["analysis"]["confidence"]["confidence_score"]
            posture = report["analysis"].get("posture", {}).get("status", "Not analyzed")
            
            if tone_score < 0:
                improvement_areas["Positive language"] = improvement_areas.get("Positive language", 0) + 1
            if filler_count > 3:
                improvement_areas["Filler words reduction"] = improvement_areas.get("Filler words reduction", 0) + 1
            if prof_word_count < 2:
                improvement_areas["Professional vocabulary"] = improvement_areas.get("Professional vocabulary", 0) + 1
            if confidence_score < 5:
                improvement_areas["Confidence building"] = improvement_areas.get("Confidence building", 0) + 1
            if posture != "Good" and posture != "Not analyzed":
                improvement_areas["Posture improvement"] = improvement_areas.get("Posture improvement", 0) + 1
        
        # Sort improvement areas by frequency
        sorted_areas = sorted(improvement_areas.items(), key=lambda x: x[1], reverse=True)
        
        if sorted_areas:
            for area, count in sorted_areas:
                percentage = (count / num_sessions) * 100
                elements.append(Paragraph(f"• {area}: Present in {count}/{num_sessions} sessions ({percentage:.1f}%)", styles['Normal']))
        else:
            elements.append(Paragraph("No consistent improvement areas identified.", styles['Normal']))
        
        elements.append(Spacer(1, 20))
        
        # Add recommendations section
        elements.append(Paragraph("Recommendations", styles['Heading2']))
        elements.append(Spacer(1, 10))
        
        recommendations = [
            "Practice answering common interview questions with a timer to build confidence.",
            "Record yourself and review for filler words and hesitations.",
            "Practice good posture in front of a mirror, keeping shoulders straight and head centered.",
            "Build a vocabulary list of professional terms relevant to your field.",
            "Practice positive framing techniques to highlight strengths even when discussing challenges."
        ]
        
        for rec in recommendations:
            elements.append(Paragraph(f"• {rec}", styles['Normal']))
    
    # Build the PDF
    doc.build(elements)
    pdf_data = buffer.getvalue()
    buffer.close()
    
    return pdf_data

def _fallback_transcript(role, experience, additional_details, interview_type):
    ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    return (
        f"Interviewer: Welcome to this {interview_type.lower()} interview for the {role} role.\n"
        f"Candidate: Thank you. My {experience.lower()} background aligns well with the position.\n"
        f"Interviewer: Could you summarize your relevant experience?\n"
        f"Candidate: {additional_details} These experiences have shaped my approach.\n"
        f"[Locally generated fallback at {ts} due to API unavailability]"
    )

def generate_interview_transcript(role, experience, additional_details, interview_type):
    model_engine = "gpt-3.5-turbo"
    prompt = (
        f"Generate a {interview_type} mock interview script to be used in a video for a {experience} {role} candidate. "
        f"Incorporate any relevant details like candidate's name, interviewer's name, company details, etc. Keep the transcript concise and focused on the interview conversation. "
        f"Additional details: {additional_details}"
    )
    messages = [
        {"role": "system", "content": "You are an assistant generating realistic mock interview transcripts."},
        {"role": "user", "content": prompt}
    ]
    if not openai_api_key:
        return _fallback_transcript(role, experience, additional_details, interview_type)
    try:
        if _OPENAI_NEW:
            response = client.chat.completions.create(
                model=model_engine,
                messages=messages,
                max_tokens=1024,
                n=1,
                temperature=0.7,
            )
            return response.choices[0].message.content
        else:
            legacy_resp = openai.ChatCompletion.create(
                model=model_engine,
                messages=messages,
                max_tokens=1024,
                n=1,
                temperature=0.7,
            )
            return legacy_resp.choices[0].message.content
    except Exception as e:
        return _fallback_transcript(role, experience, additional_details, interview_type)

def generate_audio_for_video(script, lang='en'):
    """Generate audio with different voices for interviewer and candidate."""
    lines = script.split('\n')
    audio_segments = []
    
    for line in lines:
        if ':' in line:
            speaker, text = line.split(':', 1)
            text = text.strip()
            if not text:
                continue
            
            # Use different accents for different speakers
            # Interviewer: British English (more formal)
            # Candidate: American English (more casual)
            if 'Interviewer' in speaker:
                accent = 'co.uk'  # British accent
            else:  # Candidate
                accent = 'com'    # American accent
            
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_segment:
                segment_file = temp_segment.name
                try:
                    tts = gTTS(text=text, lang=lang, tld=accent, slow=False)
                    tts.save(segment_file)
                    audio_segments.append(segment_file)
                except Exception as e:
                    # Fallback to default accent if specific accent fails
                    try:
                        tts = gTTS(text=text, lang=lang, slow=False)
                        tts.save(segment_file)
                        audio_segments.append(segment_file)
                    except:
                        st.warning(f"Audio generation warning: {e}. Continuing with available segments.")
                        if os.path.exists(segment_file):
                            os.unlink(segment_file)
                        continue
    
    if not audio_segments:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp:
            audio_file = temp.name
            tts = gTTS(text=script, lang=lang, slow=False)
            tts.save(audio_file)
        return audio_file
    
    try:
        combined = AudioSegment.empty()
        
        for segment_file in audio_segments:
            audio = AudioSegment.from_mp3(segment_file)
            combined += audio
            combined += AudioSegment.silent(duration=800)  # 800ms pause between speakers
            os.unlink(segment_file)
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp:
            audio_file = temp.name
            combined.export(audio_file, format="mp3")
        
        return audio_file
    except Exception as e:
        st.error(f"Error combining audio: {e}")
        # Fallback: return first segment if available
        if audio_segments and os.path.exists(audio_segments[0]):
            return audio_segments[0]
        # Last resort: create simple TTS
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp:
            audio_file = temp.name
            tts = gTTS(text=script, lang=lang, slow=False)
            tts.save(audio_file)
        return audio_file

def generate_image(prompt, api_key):
    """Generate realistic images using Freepik AI API."""
    if not api_key:
        st.error("Freepik API key is required for image generation")
        return None
    
    API_URL = "https://api.freepik.com/v1/ai/text-to-image"
    headers = {
        "x-freepik-api-key": api_key,
        "Content-Type": "application/json"
    }
    
    # Enhance prompt for professional interview images
    enhanced_prompt = f"{prompt}, professional photography, high quality, realistic, corporate setting, well-lit"
    
    payload = {
        "prompt": enhanced_prompt,
        "negative_prompt": "cartoon, anime, drawing, sketch, low quality, blurry",
        "guidance_scale": 7,
        "num_images": 1,
        "image": {
            "size": "landscape_16_9"
        }
    }
    
    try:
        st.info(f"🎨 Generating realistic image with Freepik AI...")
        response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            # Debug: Print response structure
            print(f"Freepik API Response keys: {result.keys()}")
            if 'data' in result:
                print(f"Data array length: {len(result.get('data', []))}")
                if len(result.get('data', [])) > 0:
                    print(f"First data item keys: {result['data'][0].keys()}")
            
            if result.get('data') and len(result['data']) > 0:
                image_data = result['data'][0]
                
                # Freepik returns base64-encoded image in 'base64' field
                if 'base64' in image_data:
                    import base64
                    # Decode base64 image
                    image_bytes = base64.b64decode(image_data['base64'])
                    img = Image.open(io.BytesIO(image_bytes))
                    st.success(f"✓ Generated realistic AI image!")
                    return img
                else:
                    st.warning(f"Base64 field not found. Available fields: {list(image_data.keys())}")
                    return None
            else:
                st.error(f"No image data in response. Response: {result}")
                return None
        else:
            error_msg = response.text[:500] if response.text else "Unknown error"
            st.error(f"Freepik AI API error ({response.status_code}): {error_msg}")
            print(f"Full error response: {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Image generation failed: {str(e)}")
        print(f"Full exception: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_video_with_images_and_audio(images, audio_file, script, output_video):
    """Create video by alternating images based on speaker with audio narration."""
    audio_clip = mp.AudioFileClip(audio_file)
    total_audio_duration = audio_clip.duration
    lines = [line for line in script.split('\n') if ':' in line and line.split(':', 1)[1].strip()]

    if not lines:
        raise ValueError("No valid dialogue lines found in transcript")

    duration_per_clip = total_audio_duration / len(lines)

    clips = []
    for i, line in enumerate(lines):
        speaker, _ = line.split(':', 1)
        # Select image based on speaker
        background = images[0] if 'Interviewer' in speaker else images[1]
        
        # Convert PIL image to numpy array and ensure RGB format
        img_array = np.array(background.convert('RGB'))
        
        # Create clip with this image
        clip = mp.ImageClip(img_array).set_duration(duration_per_clip)
        clips.append(clip)

    # Concatenate all clips and add audio
    video = mp.concatenate_videoclips(clips, method="compose")
    final_clip = video.set_audio(audio_clip)
    
    # Write video with proper settings
    final_clip.write_videofile(
        output_video, 
        codec='libx264', 
        fps=24,
        audio_codec='aac',
        temp_audiofile='temp-audio.m4a',
        remove_temp=True
    )

def create_mock_interview(role, experience, additional_details, interview_type):
    """Create mock interview video with progress updates."""
    try:
        transcript = generate_interview_transcript(role, experience, additional_details, interview_type)
        yield "✓ Generated interview transcript"

        audio_file = generate_audio_for_video(transcript)
        yield "✓ Generated audio narration"

        prompts = [
            "A professional interviewer in a modern office setting, clear and realistic.",
            "A confident candidate being interviewed, in a corporate environment, clear and realistic."
        ]

        images = []
        if not freepik_api_key:
            raise RuntimeError("FREEPIK_API_KEY is required for realistic image generation. Please set it in .env file.")
        
        yield "⏳ Generating realistic AI images with Freepik (this may take 30-60 seconds)..."
        for i, prompt in enumerate(prompts):
            image = generate_image(prompt, freepik_api_key)
            if image is None:
                raise RuntimeError(f"Failed to generate image for: {prompt}")
            images.append(image)
        yield "✓ Generated 2 realistic AI images!"

        yield "⏳ Creating video file..."
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp:
            output_video = temp.name
            create_video_with_images_and_audio(images, audio_file, transcript, output_video)
        
        yield "✓ Video creation complete!"
        yield output_video
    except Exception as e:
        yield f"❌ Error: {str(e)}"
        raise

def dashboard_page(username):
    st.title(f"Welcome to your Interview Coach Dashboard, {username}!")
    
    # Get user reports
    reports = get_user_reports(username)
    
    # Dashboard tabs
    tabs = st.tabs(["About", "Practice Interview", "Mock Interview Video", "Progress Tracker", "Reports", "Account"])
    
    # About Tab
    with tabs[0]:
        st.header("🎯 AI Interview Coach - Your Personal Interview Preparation Tool")
        
        st.markdown("""
        ### Welcome to AI Interview Coach!
        
        Our AI-powered platform helps you **master your interview skills** with cutting-edge technology and personalized feedback.
        
        ---
        
        ### 🌟 Key Features
        
        #### 🎤 **Practice Interview Modes**
        - **Audio Response Practice**: Record your answers and get AI-powered feedback on content, tone, and delivery
        - **Text Input Response**: Type your responses and receive detailed analysis
        - **Video Practice**: Practice with real-time posture detection and body language feedback using AI-powered computer vision
        
        #### 🎥 **Mock Interview Video Generator**
        - Generate **realistic AI-powered mock interview videos** with:
          - Professional interviewer images generated by Freepik AI
          - Realistic candidate images
          - Different voice accents (British for interviewer, American for candidate)
          - Customized questions based on your role and experience level
        - Download and review your mock interviews anytime
        
        #### 📊 **Progress Tracking & Analytics**
        - Track your improvement over time
        - View detailed performance metrics
        - Analyze strengths and areas for improvement
        - Visual charts showing your progress journey
        
        #### 📈 **Comprehensive Reports**
        - Detailed feedback on each practice session
        - AI-generated improvement suggestions
        - Score tracking across multiple sessions
        - Downloadable reports for your records
        
        ---
        
        ### 🔧 Technology Stack
        
        - **AI Language Model**: OpenAI GPT-3.5-turbo for intelligent feedback and interview generation
        - **Computer Vision**: MediaPipe for real-time posture detection and body language analysis
        - **Text-to-Speech**: Google gTTS with multiple accents for realistic voice generation
        - **Image Generation**: Freepik AI for creating professional, realistic interview scene images
        - **Video Processing**: MoviePy for seamless video creation and editing
        - **Web Framework**: Streamlit for an intuitive, responsive user interface
        
        ---
        
        ### 💡 How to Get Started
        
        1. **Practice Interview**: Choose your preferred practice mode and start answering questions
        2. **Mock Interview Video**: Generate a complete mock interview video with AI-generated visuals
        3. **Track Progress**: Monitor your improvement with detailed analytics
        4. **Review Reports**: Access comprehensive feedback and recommendations
        
        ---
        
        ### 🎓 Perfect For
        
        - **Job Seekers**: Prepare for upcoming interviews with confidence
        - **Students**: Practice for campus placements and internship interviews
        - **Career Changers**: Master new industry-specific interview questions
        - **Professionals**: Sharpen your interview skills for promotions or new opportunities
        
        ---
        
        ### 📞 Support
        
        Get started by selecting any tab above and begin your interview preparation journey!
        
        **Good luck with your interviews!** 🚀
        """)
        
        # Quick Stats
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Sessions", len(reports))
        
        with col2:
            avg_score = sum([r.get('score', 0) for r in reports]) / len(reports) if reports else 0
            st.metric("Avg Score", f"{avg_score:.1f}%")
        
        with col3:
            recent_sessions = len([r for r in reports if r.get('timestamp', '')])
            st.metric("Active User", "Yes" if recent_sessions > 0 else "New")
        
        with col4:
            st.metric("Account Status", "Active")
    
    # Practice Interview Tab
    with tabs[1]:
        st.header("Practice Your Interview Skills")
        
        practice_options = st.radio(
            "Choose your practice method:",
            ["Record Audio Response", "Text Input Response", "Video Practice"]
        )
        
        interview_coach = InterviewCoach()
        
        if practice_options == "Record Audio Response":
            st.subheader("Audio Response Practice")
            
            # Question selection
            questions = [
                "Tell me about yourself.",
                "What is your greatest strength?",
                "What is your greatest weakness?",
                "Why do you want to work for this company?",
                "Where do you see yourself in five years?",
                "Describe a challenging situation at work and how you handled it.",
                "Why should we hire you?",
                "What are your salary expectations?",
                "Do you have any questions for us?",
                "Add custom question..."
            ]
            
            question = st.selectbox("Select an interview question:", questions)

            # Allow user to add a custom question
            if question == "Add custom question...":
                custom_q = st.text_input("Enter your custom question:", key="custom_question_audio")
                if custom_q and custom_q.strip():
                    question = custom_q.strip()

            # Adjustable recording length
            duration_seconds = st.slider(
                "Recording duration (seconds)", min_value=5, max_value=180, value=30, step=5
            )
            interview_coach.duration = int(duration_seconds)
            
            st.write("Prepare your answer, then click 'Record' when ready.")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if st.button("Record Answer", key="start_recording"):
                    st.session_state.recording = True
                    st.info(f"Recording your answer to: {question}")
                    
                    # Create a progress bar for recording duration
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Start recording in a background thread
                    audio_queue = queue.Queue()
                    stop_event = threading.Event()
                    
                    def record_audio_thread():
                        filename = interview_coach.record_voice()
                        audio_queue.put(filename)
                    
                    recording_thread = threading.Thread(target=record_audio_thread)
                    recording_thread.start()
                    
                    # Update progress bar while recording
                    for i in range(100):
                        if stop_event.is_set():
                            break
                        progress_bar.progress(i + 1)
                        status_text.text(f"Recording: {int((i+1)/100 * interview_coach.duration)} seconds")
                        time.sleep(interview_coach.duration / 100)
                    
                    # Wait for recording to complete
                    recording_thread.join()
                    
                    # Get filename from queue
                    audio_file = audio_queue.get() if not audio_queue.empty() else None
                    
                    if audio_file:
                        st.session_state.audio_file = audio_file
                        st.session_state.transcription = interview_coach.transcribe_audio(audio_file)
                        
                        if st.session_state.transcription:
                            st.success("Recording and transcription complete!")
                            
                            # Analyze the response
                            analysis_results = interview_coach.analyze_text_input(st.session_state.transcription)
                            st.session_state.analysis_results = analysis_results
                            
                            # Store the analysis in the user's reports
                            report_data = {
                                "question": question,
                                "answer": st.session_state.transcription,
                                "analysis": analysis_results
                            }
                            
                            save_interview_report(username, report_data)
                        else:
                            st.error("No speech detected. Please try again.")
                    else:
                        st.error("Recording failed or was canceled.")
            
            with col2:
                if st.button("Stop Recording", key="stop_recording"):
                    interview_coach.stop_recording = True
                    st.info("Stopping recording...")
            
            # Display transcription and analysis if available
            if 'transcription' in st.session_state and st.session_state.transcription:
                st.subheader("Your Response")
                st.write(st.session_state.transcription)
                
                if 'analysis_results' in st.session_state:
                    analysis_results = st.session_state.analysis_results
                    feedback = interview_coach.provide_comprehensive_feedback(analysis_results)
                    
                    st.subheader("Analysis Results")
                    with st.expander("View Detailed Feedback", expanded=True):
                        st.text(feedback)
        
        elif practice_options == "Text Input Response":
            st.subheader("Text Response Practice")
            
            # Question selection
            questions = [
                "Tell me about yourself.",
                "What is your greatest strength?",
                "What is your greatest weakness?",
                "Why do you want to work for this company?",
                "Where do you see yourself in five years?",
                "Describe a challenging situation at work and how you handled it.",
                "Why should we hire you?",
                "What are your salary expectations?",
                "Do you have any questions for us?",
                "Add custom question..."
            ]
            
            question = st.selectbox("Select an interview question:", questions)

            # Allow user to add a custom question
            if question == "Add custom question...":
                custom_q = st.text_input("Enter your custom question:", key="custom_question_text")
                if custom_q and custom_q.strip():
                    question = custom_q.strip()
            
            st.write("Type your answer below:")
            user_response = st.text_area("Your answer", height=200)
            
            if st.button("Analyze Response", key="analyze_text"):
                if user_response.strip():
                    # Analyze the response
                    analysis_results = interview_coach.analyze_text_input(user_response)
                    
                    # Store the analysis in the user's reports
                    report_data = {
                        "question": question,
                        "answer": user_response,
                        "analysis": analysis_results
                    }
                    
                    save_interview_report(username, report_data)
                    
                    feedback = interview_coach.provide_comprehensive_feedback(analysis_results)
                    
                    st.subheader("Analysis Results")
                    with st.expander("View Detailed Feedback", expanded=True):
                        st.text(feedback)
                else:
                    st.warning("Please enter your response before analyzing.")
        
        elif practice_options == "Video Practice":
            st.subheader("Video Interview Practice")
            
            # Question selection
            questions = [
                "Tell me about yourself.",
                "What is your greatest strength?",
                "What is your greatest weakness?",
                "Why do you want to work for this company?",
                "Where do you see yourself in five years?",
                "Describe a challenging situation at work and how you handled it.",
                "Why should we hire you?",
                "What are your salary expectations?",
                "Do you have any questions for us?",
                "Add custom question..."
            ]
            
            question = st.selectbox("Select an interview question:", questions)

            # Allow user to add a custom question
            if question == "Add custom question...":
                custom_q = st.text_input("Enter your custom question:", key="custom_question_video")
                if custom_q and custom_q.strip():
                    question = custom_q.strip()
            
            st.write("When ready, click 'Start Camera' and answer the question while monitoring your posture.")
            st.write("The app will analyze your posture in real-time and provide feedback.")
            
            # Initialize video transformer
            video_transformer = VideoTransformer()
            
            # Add WebRTC streamer component
            webrtc_ctx = webrtc_streamer(
                key="interview-practice",
                video_transformer_factory=lambda: video_transformer,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={
                    "video": {
                        "width": {"min": 640, "ideal": 1920},
                        "height": {"min": 480, "ideal": 1080},
                        "frameRate": {"ideal": 30, "max": 30}
                    },
                    "audio": False
                },
            )
            
            # Display question prominently
            if webrtc_ctx.video_transformer:
                st.markdown(f"### Question: {question}")
                
                # Display posture status and feedback in real-time
                posture_status_ph = st.empty()
                posture_feedback_ph = st.empty()
                recommendations_ph = st.empty()
                audio_status_ph = st.empty()
                transcription_ph = st.empty()
                record_audio_ph = st.empty()

                if webrtc_ctx.state.playing:
                    posture_status_ph.markdown(f"**Current Posture Status:** {video_transformer.posture_status}")
                    posture_feedback_ph.markdown(f"**Feedback:** {video_transformer.posture_feedback}")
                    recs = video_transformer.posture_recommendations()
                    recommendations_ph.markdown("**Recommendations:**<br>" + "<br>".join([f"• {r}" for r in recs]), unsafe_allow_html=True)

                    if 'video_practice_audio_file' not in st.session_state:
                        st.session_state.video_practice_audio_file = None
                    if 'video_practice_transcription' not in st.session_state:
                        st.session_state.video_practice_transcription = ""

                    vid_duration_seconds = st.slider(
                        "Audio record length (seconds)", min_value=5, max_value=180, value=int(getattr(interview_coach, 'duration', 30)), step=5
                    )
                    interview_coach.duration = int(vid_duration_seconds)

                    if record_audio_ph.button(f"Record Audio ({interview_coach.duration}s)"):
                        audio_status_ph.info("Recording audio...")
                        audio_filename = interview_coach.record_voice("video_practice_audio.wav")
                        st.session_state.video_practice_audio_file = audio_filename
                        if audio_filename:
                            transcription = interview_coach.transcribe_audio(audio_filename)
                            st.session_state.video_practice_transcription = transcription
                            transcription_ph.success("Audio recorded and transcribed.")
                            if transcription:
                                transcription_ph.markdown(f"**Transcription:** {transcription}")
                            else:
                                transcription_ph.warning("No speech detected in audio recording.")
                        else:
                            transcription_ph.error("Audio recording failed or too short.")

                    if st.session_state.video_practice_transcription:
                        transcription_ph.markdown(f"**Transcription:** {st.session_state.video_practice_transcription}")

                    if st.session_state.video_practice_transcription.strip():
                        if 'video_practice_analyzed' not in st.session_state or not st.session_state.video_practice_analyzed:
                            analysis_results = interview_coach.analyze_text_input(st.session_state.video_practice_transcription)
                            analysis_results['posture'] = {
                                'status': video_transformer.posture_status,
                                'feedback': video_transformer.posture_feedback
                            }
                            report_data = {
                                "question": question,
                                "answer": st.session_state.video_practice_transcription,
                                "analysis": analysis_results
                            }
                            save_interview_report(username, report_data)
                            st.session_state.video_practice_feedback = interview_coach.provide_comprehensive_feedback(analysis_results)
                            st.session_state.video_practice_analyzed = True

                    if 'video_practice_feedback' in st.session_state:
                        st.subheader("Analysis Results")
                        with st.expander("View Detailed Feedback", expanded=True):
                            st.text(st.session_state.video_practice_feedback)
    
    # Mock Interview Video Tab
    with tabs[2]:
        st.header("🎬 Generate Mock Interview Video")
        st.write("Create AI-generated mock interview videos to practice your interview skills!")
        
        # Check API keys
        missing_keys = []
        if not openai_api_key:
            missing_keys.append("OPENAI_API_KEY (for transcript generation)")
        if not freepik_api_key:
            missing_keys.append("FREEPIK_API_KEY (for realistic images)")
        
        if missing_keys:
            st.error(f"⚠️ Missing required API keys: {', '.join(missing_keys)}")
            st.info("Please set these keys in your .env file to generate mock interview videos.")
        
        col1, col2 = st.columns(2)
        with col1:
            role = st.text_input("🎨 Job Role (e.g., Data Analyst, UX Designer)", "Data Analyst", key="mock_role")
            experience = st.selectbox("🌟 Experience Level", ["Entry-level", "Mid-level", "Senior", "Executive"], key="mock_exp")
        with col2:
            interview_type = st.selectbox("🎭 Interview Scenario", ["Standard", "Behavioral", "Technical", "Case Study"], key="mock_type")
            additional_details = st.text_area("✨ Additional Details", "Proficient in SQL, Python, and Tableau. The company is a growing startup in the e-commerce sector.", key="mock_details")

        if st.button("🎥 Generate Mock Interview Video"):
            progress_generator = create_mock_interview(role, experience, additional_details, interview_type)
            progress_placeholder = st.empty()
            
            for progress_text in progress_generator:
                if progress_text.endswith(".mp4"):
                    st.success("🌟 Your mock interview video is ready!")
                    st.video(progress_text)
                    with open(progress_text, 'rb') as f:
                        st.download_button(
                            label="📥 Download Video",
                            data=f,
                            file_name=f"{username}_mock_interview.mp4",
                            mime="video/mp4",
                            key="download_mock_video"
                        )
                    # Clean up temp file
                    try:
                        os.unlink(progress_text)
                    except:
                        pass
                else:
                    progress_placeholder.write(progress_text)
        
        st.markdown("---")
        st.markdown("### How It Works")
        st.write("""
        1. 🤖 GPT-3.5 generates a custom interview transcript based on your role and experience
        2. 🎙️ Google TTS creates audio with different voices (British accent for interviewer, American accent for candidate)
        3. 🖼️ Freepik AI generates ultra-realistic professional interviewer and candidate images
        4. 🎬 MoviePy assembles everything into a high-quality mock interview video!
        """)
    
    # Progress Tracker Tab
    with tabs[3]:
        st.header("Your Interview Progress")
        
        if not reports:
            st.info("You haven't completed any practice interviews yet. Start practicing to see your progress!")
        else:
            # Prepare data for charts
            dates = []
            confidence_scores = []
            tone_scores = []
            filler_counts = []
            professional_counts = []
            
            for report in reports:
                dates.append(report["timestamp"])
                confidence_scores.append(report["analysis"]["confidence"]["confidence_score"])
                tone_scores.append(report["analysis"]["tone"]["score"])
                filler_counts.append(report["analysis"]["word_choice"]["filler_word_count"])
                professional_counts.append(report["analysis"]["word_choice"]["professional_word_count"])
            
            # Convert to pandas for easier charting
            df = pd.DataFrame({
                "Date": dates,
                "Confidence": confidence_scores,
                "Tone": tone_scores,
                "Filler Words": filler_counts,
                "Professional Words": professional_counts
            })
            
            # Show total number of practice sessions
            st.subheader(f"Total Practice Sessions: {len(reports)}")
            
            # Create multiple charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Confidence Score Over Time")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(df["Date"], df["Confidence"], marker='o', linestyle='-', color='blue')
                ax.set_ylim(0, 10)
                ax.set_ylabel("Confidence Score (0-10)")
                ax.set_xlabel("Session")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                st.subheader("Professional vs. Filler Words")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(df["Date"], df["Professional Words"], label="Professional Words", color='green')
                ax.bar(df["Date"], df["Filler Words"], label="Filler Words", color='red', alpha=0.7)
                ax.set_ylabel("Word Count")
                ax.set_xlabel("Session")
                ax.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                st.subheader("Tone Score Over Time")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(df["Date"], df["Tone"], marker='o', linestyle='-', color='green')
                ax.set_ylim(-1, 1)
                ax.set_ylabel("Tone Score (-1 to 1)")
                ax.set_xlabel("Session")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Calculate average scores
                avg_confidence = df["Confidence"].mean()
                avg_tone = df["Tone"].mean()
                avg_filler = df["Filler Words"].mean()
                avg_professional = df["Professional Words"].mean()
                
                # Show average scores
                st.subheader("Average Scores")
                col1, col2 = st.columns(2)
                col1.metric("Avg Confidence", f"{avg_confidence:.1f}/10")
                col2.metric("Avg Tone", f"{avg_tone:.2f}")
                col1.metric("Avg Filler Words", f"{avg_filler:.1f}")
                col2.metric("Avg Professional Words", f"{avg_professional:.1f}")
    
    # Reports Tab
    with tabs[4]:
        st.header("Your Interview Reports")
        
        if not reports:
            st.info("You haven't completed any practice interviews yet. Start practicing to generate reports!")
        else:
            # Date range filter
            st.subheader("Filter Reports")
            col1, col2 = st.columns(2)
            
            # Get min and max dates from reports
            dates = [datetime.strptime(report["date"], "%Y-%m-%d") for report in reports]
            min_date = min(dates).date()
            max_date = max(dates).date()
            
            with col1:
                start_date = st.date_input("Start Date", min_date)
            
            with col2:
                end_date = st.date_input("End Date", max_date)
            
            # Filter reports by date
            filtered_reports = [r for r in reports if start_date.strftime("%Y-%m-%d") <= r["date"] <= end_date.strftime("%Y-%m-%d")]
            
            if not filtered_reports:
                st.warning("No reports found in the selected date range.")
            else:
                st.subheader(f"Found {len(filtered_reports)} reports in selected date range")
                
                # Generate PDF report button
                if st.button("Generate PDF Report"):
                    pdf_data = create_progress_report_pdf(username, filtered_reports, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_data,
                        file_name=f"{username}_interview_report_{start_date}_to_{end_date}.pdf",
                        mime="application/pdf"
                    )
                
                # Show individual reports
                for i, report in enumerate(filtered_reports):
                    with st.expander(f"Report {i+1}: {report['timestamp']} - {report['question'][:30]}..."):
                        st.markdown(f"**Question:** {report['question']}")
                        st.markdown(f"**Your Answer:** {report['answer']}")
                        
                        # Display analysis summary
                        st.markdown("### Analysis Summary")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Confidence Score", f"{report['analysis']['confidence']['confidence_score']:.1f}/10")
                        col2.metric("Tone Score", f"{report['analysis']['tone']['score']:.2f}")
                        col3.metric("Filler Words", f"{report['analysis']['word_choice']['filler_word_count']}")
                        
                        # Display full feedback
                        st.markdown("### Detailed Feedback")
                        feedback = interview_coach.provide_comprehensive_feedback(report['analysis'])
                        st.text(feedback)
    
    # Account Tab
    with tabs[5]:
        st.header("Account Settings")
        
        st.subheader("User Information")
        st.write(f"Username: {username}")
        st.write(f"Account Created: Unknown")  # Could store creation date in user DB
        st.write(f"Total Practice Sessions: {len(reports)}")
        
        if st.button("Log Out"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.success("You have been logged out successfully.")
            st.rerun()

def main():
    # Initialize database
    init_db()
    
    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    if 'username' not in st.session_state:
        st.session_state.username = None
    
    if 'show_login' not in st.session_state:
        st.session_state.show_login = True
    
    # App title
    st.title("Interview Coach AI")
    st.write("Practice and improve your interview skills with AI feedback")
    
    if st.session_state.logged_in:
        dashboard_page(st.session_state.username)
    else:
        # Authentication section
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Login" if not st.session_state.show_login else "Already have an account? Login"):
                st.session_state.show_login = True
        
        with col2:
            if st.button("Sign Up" if st.session_state.show_login else "Need an account? Sign Up"):
                st.session_state.show_login = False
        
        if st.session_state.show_login:
            create_login_page()
        else:
            create_signup_page()

if __name__ == "__main__":
    main()