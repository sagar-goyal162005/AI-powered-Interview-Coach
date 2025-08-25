import streamlit as st
import speech_recognition as sr
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import numpy as np
import sounddevice as sd
import wavio
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
import io
from datetime import datetime

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)

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
            return "Cannot detect face", 0, "Unknown", img
        
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
        """Record audio"""
        self.recording_in_progress = True
        self.stop_recording = False
        
        print(f"Recording your answer for {self.duration} seconds...")
        print("Speak now!")
        
        # Calculate total samples
        total_samples = int(self.duration * self.sample_rate)
        
        # Create array to store audio data
        recording = np.zeros((total_samples, 1))
        
        # Record in chunks to allow for stopping
        chunk_size = int(0.1 * self.sample_rate)  # 0.1 second chunks
        recorded_samples = 0
        
        # Start recording
        stream = sd.InputStream(samplerate=self.sample_rate, channels=1)
        stream.start()
        
        while recorded_samples < total_samples and not self.stop_recording:
            # Calculate remaining samples in this chunk
            samples_to_record = min(chunk_size, total_samples - recorded_samples)
            
            # Record chunk
            data, overflowed = stream.read(samples_to_record)
            
            # Store in our recording array
            recording[recorded_samples:recorded_samples + len(data)] = data
            
            # Update position
            recorded_samples += len(data)
            
        # Stop and close the stream
        stream.stop()
        stream.close()
        
        # If we have any recorded data and not just stopped immediately
        if recorded_samples > self.sample_rate * 0.5:  # At least half a second
            # Trim recording to actual length
            recording = recording[:recorded_samples]
            
            # Save recording
            wavio.write(filename, recording, self.sample_rate, sampwidth=2)
            print(f"Recording saved to {filename}")
            return filename
        else:
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

def dashboard_page(username):
    st.title(f"Welcome to your Interview Coach Dashboard, {username}!")
    
    # Get user reports
    reports = get_user_reports(username)
    
    # Dashboard tabs
    tabs = st.tabs(["Practice Interview", "Progress Tracker", "Reports", "Account"])
    
    # Practice Interview Tab
    with tabs[0]:
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
                "Do you have any questions for us?"
            ]
            
            question = st.selectbox("Select an interview question:", questions)
            
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
                "Do you have any questions for us?"
            ]
            
            question = st.selectbox("Select an interview question:", questions)
            
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
                "Do you have any questions for us?"
            ]
            
            question = st.selectbox("Select an interview question:", questions)
            
            st.write("When ready, click 'Start Camera' and answer the question while monitoring your posture.")
            st.write("The app will analyze your posture in real-time and provide feedback.")
            
            # Initialize video transformer
            video_transformer = VideoTransformer()
            
            # Add WebRTC streamer component
            webrtc_ctx = webrtc_streamer(
                key="interview-practice",
                video_transformer_factory=lambda: video_transformer,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={"video": True, "audio": True},
            )
            
            # Display question prominently
            if webrtc_ctx.video_transformer:
                st.markdown(f"### Question: {question}")
                
                # Display posture status and feedback in real-time
                posture_status = st.empty()
                feedback_text = st.empty()
                
                if webrtc_ctx.state.playing:
                    posture_status.markdown(f"**Current Posture Status:** {video_transformer.posture_status}")
                    feedback_text.markdown(f"**Feedback:** {video_transformer.posture_feedback}")
                    
                    # Add text input for written answer
                    st.write("Type your answer as you speak:")
                    video_response = st.text_area("Your answer (while on camera)", height=150)
                    
                    if st.button("Submit Video Practice"):
                        if video_response.strip():
                            # Analyze the text response
                            analysis_results = interview_coach.analyze_text_input(video_response)
                            
                            # Add posture analysis
                            analysis_results['posture'] = {
                                'status': video_transformer.posture_status,
                                'feedback': video_transformer.posture_feedback
                            }
                            
                            # Store the analysis in the user's reports
                            report_data = {
                                "question": question,
                                "answer": video_response,
                                "analysis": analysis_results
                            }
                            
                            save_interview_report(username, report_data)
                            
                            feedback = interview_coach.provide_comprehensive_feedback(analysis_results)
                            
                            st.subheader("Analysis Results")
                            with st.expander("View Detailed Feedback", expanded=True):
                                st.text(feedback)
                        else:
                            st.warning("Please enter your response before submitting.")
    
    # Progress Tracker Tab
    with tabs[1]:
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
    with tabs[2]:
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
    with tabs[3]:
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