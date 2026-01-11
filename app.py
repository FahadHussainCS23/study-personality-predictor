import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import time
import os
import plotly.graph_objects as go
import streamlit.components.v1 as components
from fpdf import FPDF  # Make sure to: pip install fpdf

# --- 1. PAGE CONFIG & STYLING ---
st.set_page_config(page_title="StudyPredict AI", page_icon="🎓", layout="wide")


def style_app():
    st.markdown("""
        <style>
        .stApp { background-color: #ffffff; }
        h1, h2, h3, p, span, label, .stMarkdown { color: #1f2937 !important; }

        .header-container {
            padding: 2.5rem;
            background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
            border-radius: 15px;
            text-align: center;
            margin-bottom: 2rem;
            color: white !important;
        }

        .header-container h1, .header-container p { color: #ffffff !important; }

        .custom-card {
            background: #ffffff;
            padding: 2rem;
            border-radius: 15px;
            border: 1px solid #e5e7eb;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        }

        .inner-header {
            font-size: 1.4rem;
            font-weight: 800;
            color: #1e3a8a;
            margin-bottom: 1rem;
            display: block;
            border-bottom: 2px solid #3b82f6;
            padding-bottom: 8px;
        }

        div.stButton > button {
            border-radius: 8px;
            background-color: #3b82f6;
            color: white !important;
            border: none;
            font-weight: bold;
            height: 3em;
        }
        </style>
    """, unsafe_allow_html=True)


def force_top():
    components.html(
        """<script>
            var mainContent = window.parent.document.querySelector('section.main');
            if (mainContent) { mainContent.scrollTo({top: 0, behavior: 'auto'}); }
            window.parent.window.scrollTo(0,0);
        </script>""", height=0,
    )


style_app()

# --- 2. SESSION STATE MANAGEMENT ---
if "page" not in st.session_state:
    st.session_state.page = "landing"
if "results" not in st.session_state:
    st.session_state.results = None
if "pdf_blob" not in st.session_state:
    st.session_state.pdf_blob = None


# --- 3. PDF HELPER FUNCTION ---
def create_pdf(text, techniques):
    try:
        pdf = FPDF()
        pdf.add_page()
        
        # Title
        pdf.set_font("Arial", 'B', size=16)
        pdf.cell(200, 10, txt="Your StudyPredict AI Roadmap", ln=True, align='C')
        
        # Techniques
        pdf.set_font("Arial", 'B', size=12)
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Recommended Techniques: {techniques}", ln=True)
        
        # Body text cleaning
        pdf.set_font("Arial", size=11)
        pdf.ln(5)
        
        # Remove markdown characters that Arial can't render
        clean_text = text.replace("**", "").replace("###", "").replace("- ", "* ")
        # Convert to Latin-1, replacing emojis/special chars with '?'
        clean_text = clean_text.encode('latin-1', 'replace').decode('latin-1')
        
        pdf.multi_cell(0, 8, txt=clean_text)
        
        # Get the output
        pdf_output = pdf.output(dest='S')
        
        # FIX: Check if output is already bytes, if not, encode it
        if isinstance(pdf_output, str):
            return pdf_output.encode('latin-1')
        return bytes(pdf_output)
        
    except Exception as e:
        return f"PDF Error: {str(e)}".encode('latin-1')
# --- 4. CORE LOGIC (ML & AI) ---
@st.cache_resource
def load_ml_assets():
    try:
        if os.path.exists('personality_model.pkl') and os.path.exists('scaler.pkl'):
            with open('personality_model.pkl', 'rb') as f:
                model = pickle.load(f)
            with open('scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            return model, scaler
    except Exception as e:
        st.error(f"Error loading files: {e}")
    return None, None


def get_ai_advice(profile, techniques, subjects):
    api_key = "AIzaSyDNde4TB0jbK7Ofqs4xSsV4ROfzvWWVKTM"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={api_key}"

    prompt = f"""
    Act as a professional Study Success Coach.
    Explain why these study techniques ({techniques}) fit a student with these traits ({profile}). 
    Then, create a 7-day study schedule for these subjects: {subjects}.
    Format the output beautifully using Markdown.
    """
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
    except:
        pass
    return "AI Insight currently unavailable, but check your model results below!"


model, scaler = load_ml_assets()

# --- PAGE 1: LANDING ---
if st.session_state.page == "landing":
    force_top()
    st.markdown("""
        <div class="header-container">
            <h1>🎓 StudyPredict AI</h1>
            <p>Final Project by Arpita Lakhisirani & Fahad Hussain</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="custom-card" style="text-align:center;">', unsafe_allow_html=True)
        st.write("Unlock your learning potential with ML-driven personality analysis and AI-crafted study roadmaps.")
        if st.button("🚀 Start Assessment", use_container_width=True):
            st.session_state.page = "app"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE 2: MAIN APP ---
elif st.session_state.page == "app":
    force_top()
    st.markdown('<div class="header-container"><h2>🧠 Assessment Phase</h2></div>', unsafe_allow_html=True)

    if model is None or scaler is None:
        st.error("⚠️ Model files missing! Place .pkl files in the directory.")
    else:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        subjects = st.text_input("What are you studying?", "Data Science, Python")

        c1, c2 = st.columns(2)
        with c1:
            o = st.slider("Openness (Creativity)", 1, 5, 3)
            c = st.slider("Conscientiousness (Discipline)", 1, 5, 3)
            e = st.slider("Extraversion (Social Energy)", 1, 5, 3)
        with c2:
            a = st.slider("Agreeableness (Cooperation)", 1, 5, 3)
            n = st.slider("Neuroticism (Stress Sensitivity)", 1, 5, 3)

        if st.button("✨ Architect My Plan", use_container_width=True):
            with st.spinner("Processing..."):
                input_data = np.array([[o, c, e, a, n]])
                scaled_input = scaler.transform(input_data)
                prediction = model.predict(scaled_input)[0]
                tech_names = ['Spaced Repetition', 'Active Recall', 'Feynman Technique', 'Mind Mapping',
                              'Pomodoro Technique']
                results = [tech_names[i] for i, val in enumerate(prediction) if val == 1]
                if not results: results = ['Active Recall']

                ai_text = get_ai_advice(f"O:{o}, C:{c}, E:{e}, A:{a}, N:{n}", ", ".join(results), subjects)

                st.session_state.results = {"tech": results, "ai": ai_text,
                                            "scores": {"Openness": o, "Conscientiousness": c, "Extraversion": e,
                                                       "Agreeableness": a, "Neuroticism": n}}
                st.session_state.page = "results"
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE 3: RESULTS & PDF ---
elif st.session_state.page == "results":
    force_top()
    st.markdown('<div class="header-container"><h2>📊 Your Study Blueprint</h2></div>', unsafe_allow_html=True)

    res = st.session_state.results
    col_l, col_r = st.columns([1, 1.5], gap="large")

    with col_l:
        st.markdown('<div class="custom-card"><span class="inner-header">📊 Visual Profile</span>',
                    unsafe_allow_html=True)
        # Radar Chart
        fig = go.Figure(data=go.Scatterpolar(
            r=list(res["scores"].values()) + [list(res["scores"].values())[0]],
            theta=list(res["scores"].keys()) + [list(res["scores"].keys())[0]],
            fill='toself', fillcolor='rgba(59, 130, 246, 0.4)', line_color='#1e3a8a'
        ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5])), showlegend=False,
                          margin=dict(l=40, r=40, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="custom-card"><span class="inner-header">✨ AI Personalized Roadmap</span>',
                    unsafe_allow_html=True)
        st.markdown(res["ai"])

        st.divider()

        # PDF DOWNLOAD LOGIC
        if st.session_state.pdf_blob is None:
            if st.button("🛠️ Prepare PDF Report", use_container_width=True):
                st.session_state.pdf_blob = create_pdf(res["ai"], ", ".join(res["tech"]))
                st.rerun()
        else:
            st.download_button(
                label="📥 Download Study Plan PDF",
                data=st.session_state.pdf_blob,
                file_name="My_Study_Plan.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            if st.button("🗑️ Clear PDF Cache"):
                st.session_state.pdf_blob = None
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔄 Restart"):
        st.session_state.page = "landing"
        st.session_state.results = None
        st.session_state.pdf_blob = None

        st.rerun()

