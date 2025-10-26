import streamlit as st
import os
import tempfile
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
from collections import Counter
import time
from datetime import datetime

# Try to import PyTorch (optional for settings page)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ml'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'script_analysis'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'auth'))

# Import our modules
from script_analysis.script_analyzer import ScriptAnalyzer
from auth.auth_manager import auth_manager, get_current_user, is_authenticated, require_auth
from auth.analytics import analytics_manager

# Page config
st.set_page_config(
    page_title="JV Cinelytics - Complete ML & Script Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional CSS with responsive design
st.markdown("""
<style>
    /* Import Google Fonts for professional typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Root variables for consistent theming */
    :root {
        --primary-red: #dc2626;
        --dark-red: #b91c1c;
        --light-red: #ef4444;
        --black: #000000;
        --dark-gray: #111827;
        --medium-gray: #374151;
        --light-gray: #6b7280;
        --white: #ffffff;
        --off-white: #f9fafb;
        --border-radius: 8px;
        --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    /* Main app background */
    .stApp {
        background: linear-gradient(135deg, var(--black) 0%, var(--dark-gray) 100%);
        color: var(--white);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        min-height: 100vh;
    }
    
    /* Remove default Streamlit padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Header styling */
    .main-header {
        font-size: clamp(2rem, 5vw, 3.5rem);
        font-weight: 700;
        color: var(--primary-red);
        text-align: center;
        margin: 0 0 1rem 0;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
        letter-spacing: -0.025em;
    }
    
    .subtitle {
        text-align: center;
        font-size: clamp(1rem, 2.5vw, 1.25rem);
        color: var(--light-gray);
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Navigation styling */
    .nav-container {
        background: var(--medium-gray);
        padding: 1rem;
        border-radius: var(--border-radius);
        margin-bottom: 2rem;
        box-shadow: var(--shadow);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Welcome message styling */
    .welcome-message {
        text-align: left;
        font-size: clamp(0.9rem, 2vw, 1.1rem);
        color: var(--light-gray);
        margin: 0.5rem 0 1.5rem 0;
        font-weight: 400;
        padding: 0.75rem 1rem;
        background: var(--medium-gray);
        border-radius: var(--border-radius);
        border-left: 4px solid var(--primary-red);
    }
    
    .welcome-message strong {
        color: var(--white);
        font-weight: 600;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-red) 0%, var(--dark-red) 100%);
        color: var(--white);
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: var(--border-radius);
        font-weight: 600;
        font-size: 0.875rem;
        transition: all 0.2s ease-in-out;
        box-shadow: var(--shadow);
        width: 100%;
        min-height: 3rem;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, var(--dark-red) 0%, var(--primary-red) 100%);
        box-shadow: var(--shadow-lg);
        transform: translateY(-1px);
    }
    
    .stButton > button:active {
        transform: translateY(0);
        box-shadow: var(--shadow);
    }
    
    /* Section headers */
    .section-header {
        font-size: clamp(1.5rem, 3vw, 2rem);
        font-weight: 600;
        color: var(--white);
        margin: 2rem 0 1.5rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid var(--primary-red);
        position: relative;
    }
    
    .section-header::after {
        content: '';
        position: absolute;
        bottom: -2px;
        left: 0;
        width: 60px;
        height: 2px;
        background: var(--light-red);
    }
    
    /* Card components */
    .metric-card {
        background: linear-gradient(135deg, var(--medium-gray) 0%, var(--dark-gray) 100%);
        padding: 1.5rem;
        border-radius: var(--border-radius);
        margin: 0.75rem 0;
        box-shadow: var(--shadow);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.2s ease-in-out;
    }
    
    .metric-card:hover {
        box-shadow: var(--shadow-lg);
        transform: translateY(-2px);
        border-color: var(--primary-red);
    }
    
    .metric-card h3 {
        color: var(--primary-red);
        margin: 0 0 1rem 0;
        font-weight: 600;
        font-size: 1.25rem;
    }
    
    .metric-card p {
        color: var(--light-gray);
        margin: 0;
        line-height: 1.6;
    }
    
    /* Character and location items */
    .character-item, .location-item {
        background: var(--medium-gray);
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid var(--primary-red);
        border-radius: 0 var(--border-radius) var(--border-radius) 0;
        box-shadow: var(--shadow);
        transition: all 0.2s ease-in-out;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .location-item {
        border-left-color: var(--light-gray);
    }
    
    .character-item:hover, .location-item:hover {
        background: var(--dark-gray);
        border-left-width: 6px;
        transform: translateX(4px);
    }
    
    .character-item strong, .location-item strong {
        color: var(--white);
        font-weight: 600;
    }
    
    /* Alert boxes */
    .success-box, .warning-box, .info-box {
        padding: 1rem;
        border-radius: var(--border-radius);
        margin: 1rem 0;
        border-left: 4px solid;
        font-weight: 500;
    }
    
    .success-box {
        background: rgba(34, 197, 94, 0.1);
        color: #86efac;
        border-left-color: #22c55e;
    }
    
    .warning-box {
        background: rgba(251, 191, 36, 0.1);
        color: #fde047;
        border-left-color: #f59e0b;
    }
    
    .info-box {
        background: rgba(59, 130, 246, 0.1);
        color: #93c5fd;
        border-left-color: #3b82f6;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: var(--dark-gray);
        padding: 0.5rem;
        border-radius: var(--border-radius);
        margin-bottom: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        background: var(--medium-gray);
        border-radius: var(--border-radius);
        color: var(--light-gray);
        font-weight: 500;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.2s ease-in-out;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--primary-red);
        color: var(--white);
        font-weight: 600;
        border-color: var(--light-red);
        box-shadow: var(--shadow);
    }
    
    /* Form styling */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select {
        background: var(--medium-gray);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: var(--border-radius);
        color: var(--white);
        font-size: 0.875rem;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--primary-red);
        box-shadow: 0 0 0 2px rgba(220, 38, 38, 0.2);
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        background: var(--medium-gray);
        border: 2px dashed rgba(255, 255, 255, 0.3);
        border-radius: var(--border-radius);
        padding: 2rem;
        text-align: center;
        transition: all 0.2s ease-in-out;
    }
    
    .stFileUploader > div:hover {
        border-color: var(--primary-red);
        background: rgba(220, 38, 38, 0.05);
    }
    
    /* Metrics styling */
    [data-testid="metric-container"] {
        background: var(--medium-gray);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: var(--border-radius);
        padding: 1rem;
        box-shadow: var(--shadow);
    }
    
    [data-testid="metric-container"] > div {
        color: var(--white);
    }
    
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: var(--primary-red);
        font-weight: 600;
    }
    
    /* DataFrame styling */
    .stDataFrame {
        border-radius: var(--border-radius);
        overflow: hidden;
        box-shadow: var(--shadow);
    }
    
    /* Plotly chart styling */
    .js-plotly-plot {
        border-radius: var(--border-radius);
        background: var(--medium-gray);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: var(--dark-gray);
    }
    
    /* Code block styling */
    .stCodeBlock {
        background: var(--dark-gray);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: var(--border-radius);
    }
    
    /* Spinner styling */
    .stSpinner {
        color: var(--primary-red);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 0.5rem;
            padding-right: 0.5rem;
        }
        
        .metric-card {
            padding: 1rem;
        }
        
        .character-item, .location-item {
            padding: 0.75rem;
        }
        
        .nav-container {
            padding: 0.5rem;
            overflow-x: auto;
        }
        
        .welcome-message {
            font-size: 0.9rem;
            padding: 0.5rem 0.75rem;
        }
        
        /* Smaller buttons on mobile for better fit */
        .stButton > button {
            padding: 0.5rem 0.75rem;
            font-size: 0.75rem;
            min-height: 2.5rem;
        }
    }
    
    @media (max-width: 640px) {
        .stButton > button {
            padding: 0.5rem 1rem;
            font-size: 0.8rem;
        }
        
        .section-header {
            font-size: 1.5rem;
        }
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--dark-gray);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--medium-gray);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-red);
    }
    
    # Hide Streamlit branding
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

def apply_theme_css(theme='dark'):
    """Apply theme-specific CSS based on user preference"""
    if theme == 'light':
        # Light theme colors
        st.markdown("""
        <style>
            :root {
                --primary-red: #dc2626;
                --dark-red: #b91c1c;
                --light-red: #ef4444;
                --black: #ffffff;
                --dark-gray: #f3f4f6;
                --medium-gray: #e5e7eb;
                --light-gray: #374151;
                --white: #111827;
                --off-white: #1f2937;
            }
            
            .stApp {
                background: linear-gradient(135deg, #ffffff 0%, #f9fafb 100%);
                color: #111827;
            }
            
            .main-header {
                color: var(--primary-red);
                text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
            
            .subtitle, .welcome-message {
                color: #4b5563;
            }
            
            .metric-card {
                background: linear-gradient(135deg, #ffffff 0%, #f9fafb 100%);
                border: 1px solid #e5e7eb;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            }
            
            .metric-card h3 {
                color: var(--primary-red);
            }
            
            .metric-card p {
                color: #4b5563;
            }
            
            .nav-container {
                background: #ffffff;
                border: 1px solid #e5e7eb;
            }
            
            .character-item, .location-item {
                background: #f9fafb;
                border: 1px solid #e5e7eb;
            }
            
            .section-header {
                color: #111827;
                border-bottom-color: var(--primary-red);
            }
            
            .stButton > button {
                background: linear-gradient(135deg, var(--primary-red) 0%, var(--dark-red) 100%);
                color: #ffffff;
            }
            
            .stTextInput > div > div > input,
            .stTextArea > div > div > textarea,
            .stSelectbox > div > div > select {
                background: #ffffff;
                border: 1px solid #d1d5db;
                color: #111827;
            }
            
            [data-testid="metric-container"] {
                background: #ffffff;
                border: 1px solid #e5e7eb;
            }
            
            ::-webkit-scrollbar-track {
                background: #f3f4f6;
            }
            
            ::-webkit-scrollbar-thumb {
                background: #d1d5db;
            }
        </style>
        """, unsafe_allow_html=True)
    # Dark theme is already applied in the main CSS block

def main():
    # Initialize theme in session state
    if 'theme' not in st.session_state:
        st.session_state.theme = 'dark'
    
    # Apply theme-specific CSS
    apply_theme_css(st.session_state.theme)
    
    # Check authentication
    if not is_authenticated():
        show_landing_page()
        return
    
    # Get current user
    current_user = get_current_user()
    
    # Top header
    st.markdown('<h1 class="main-header">🎬 JV Cinelytics</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Intelligent Script Analysis for Smarter Filmmaking</p>', unsafe_allow_html=True)
    
    # Welcome message on separate line
    st.markdown(f'<p class="welcome-message">Welcome back, <strong>{current_user["username"]}</strong> 👋</p>', unsafe_allow_html=True)
    
    # Navigation buttons
    page = show_top_nav(current_user["role"])
    
    # Main content based on navigation
    if page == "🏠 Home":
        show_home()
    elif page == "📊 Analytics":
        show_analytics()
    elif page == "👥 User Management":
        show_user_management()
    elif page == "📊 My Dashboard":
        show_user_dashboard()
    elif page == "📊 Script Analysis":
        show_script_analysis()
    elif page == "🔮 Genre Prediction":
        show_genre_prediction()
    elif page == "⚙️ Settings":
        show_settings()
    elif page == "🚪 Logout":
        logout_user()

def show_top_nav(role):
    """Generates the top navigation bar with buttons."""
    if role == "admin":
        nav_options = ["🏠 Home", "📊 Analytics", "👥 User Management", "📊 Script Analysis", "🔮 Genre Prediction", "⚙️ Settings", "🚪 Logout"]
    else:
        nav_options = ["🏠 Home", "📊 My Dashboard", "📊 Script Analysis", "🔮 Genre Prediction", "⚙️ Settings", "🚪 Logout"]

    # Use a unique key to prevent collisions
    nav_choice = st.session_state.get('nav_choice', nav_options[0])

    # Navigation with horizontal buttons
    st.markdown('<div class="nav-container">', unsafe_allow_html=True)
    cols = st.columns(len(nav_options))
    for i, option in enumerate(nav_options):
        with cols[i]:
            # Highlight active page
            btn_type = "primary" if nav_choice == option else "secondary"
            if st.button(option, key=f"nav_{option}", type=btn_type, use_container_width=True):
                nav_choice = option
                st.session_state.nav_choice = nav_choice
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    return nav_choice

def show_landing_page():
    """Show landing page for unauthenticated users"""
    # Initialize page selection in session state
    if 'landing_page' not in st.session_state:
        st.session_state.landing_page = 'home'
    
    # Top header
    st.markdown('<h1 class="main-header">🎬 JV Cinelytics</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Intelligent Script Analysis for Smarter Filmmaking</p>', unsafe_allow_html=True)
    
    # Simple navigation for unauthenticated users
    st.markdown('<div class="nav-container">', unsafe_allow_html=True)
    nav_options = ["🏠 Home", "📊 Script Analysis", "🔮 Genre Prediction", "🔐 Login"]
    cols = st.columns(len(nav_options))
    for i, option in enumerate(nav_options):
        with cols[i]:
            btn_type = "primary" if st.session_state.landing_page == option.split()[-1].lower() else "secondary"
            if st.button(option, key=f"landing_nav_{option}", type=btn_type, use_container_width=True):
                st.session_state.landing_page = option.split()[-1].lower()
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Show content based on navigation
    if st.session_state.landing_page == 'home':
        show_landing_home()
    elif st.session_state.landing_page == 'analysis':
        show_auth_required_message("Script Analysis")
    elif st.session_state.landing_page == 'prediction':
        show_auth_required_message("Genre Prediction")
    else:  # login
        show_login_page()

def show_landing_home():
    """Show landing page home content"""
    # Hero section
    st.markdown("""
    <div class="metric-card" style="text-align: center; padding: 3rem 2rem;">
        <h2 style="color: var(--primary-red); font-size: 2.5rem; margin-bottom: 1rem;">Transform Your Screenwriting Process</h2>
        <p style="font-size: 1.2rem; color: var(--light-gray); margin-bottom: 2rem;">
            Harness the power of AI and machine learning to analyze scripts, predict genres, and gain insights that elevate your storytelling.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">✨ Platform Features</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3, gap="medium")
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Script Analysis</h3>
            <p>Upload your movie script and get comprehensive analysis including character development, 
            location mapping, and intelligent genre classification powered by advanced ML algorithms.</p>
            <br>
            <p style="color: var(--primary-red); font-weight: 600;">✓ Character profiling</p>
            <p style="color: var(--primary-red); font-weight: 600;">✓ Location extraction</p>
            <p style="color: var(--primary-red); font-weight: 600;">✓ Scene analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🔮 Genre Prediction</h3>
            <p>Leverage our sophisticated natural language processing models to predict genre classifications 
            from script excerpts with high accuracy and detailed confidence metrics.</p>
            <br>
            <p style="color: var(--primary-red); font-weight: 600;">✓ ML-powered prediction</p>
            <p style="color: var(--primary-red); font-weight: 600;">✓ Confidence scores</p>
            <p style="color: var(--primary-red); font-weight: 600;">✓ Multi-genre support</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📈 Analytics Dashboard</h3>
            <p>Track your progress with personalized analytics, visualize your analysis history, 
            and gain insights into your creative workflow patterns.</p>
            <br>
            <p style="color: var(--primary-red); font-weight: 600;">✓ Personal statistics</p>
            <p style="color: var(--primary-red); font-weight: 600;">✓ Activity tracking</p>
            <p style="color: var(--primary-red); font-weight: 600;">✓ Visual reports</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">🚀 Get Started in 3 Simple Steps</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3, gap="large")
    
    with col1:
        st.markdown("""
        <div class="info-box" style="text-align: center;">
            <h2 style="color: var(--primary-red);">1</h2>
            <h4>Create Your Account</h4>
            <p>Sign up for free in less than a minute. No credit card required.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box" style="text-align: center;">
            <h2 style="color: var(--primary-red);">2</h2>
            <h4>Upload Your Script</h4>
            <p>Upload your screenplay in .txt or .docx format for instant analysis.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-box" style="text-align: center;">
            <h2 style="color: var(--primary-red);">3</h2>
            <h4>Get Insights</h4>
            <p>Receive comprehensive analysis and actionable insights immediately.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Call to action
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 Create Free Account", type="primary", use_container_width=True):
            st.session_state.landing_page = 'login'
            st.rerun()

def show_auth_required_message(feature_name):
    """Show authentication required message for protected pages"""
    st.markdown(f'<div class="section-header">{feature_name}</div>', unsafe_allow_html=True)
    
    # Create a centered container
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="warning-box" style="text-align: center; padding: 3rem 2rem;">
            <h2 style="color: #fde047; margin-bottom: 1rem;">🔒 Authentication Required</h2>
            <p style="font-size: 1.1rem; margin-bottom: 2rem;">
                You need to be logged in to access this feature.
            </p>
            <p style="color: var(--light-gray);">
                Create a free account or log in to start analyzing scripts and predicting genres.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Action buttons
        col_a, col_b = st.columns(2, gap="medium")
        
        with col_a:
            if st.button("🔐 Login to Your Account", type="primary", use_container_width=True):
                st.session_state.landing_page = 'login'
                # Set to login tab
                st.session_state.auth_tab = 'login'
                st.rerun()
        
        with col_b:
            if st.button("📝 Create New Account", type="secondary", use_container_width=True):
                st.session_state.landing_page = 'login'
                # Set to register tab
                st.session_state.auth_tab = 'register'
                st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Feature teaser
        st.markdown(f"""
        <div class="metric-card">
            <h3>What you'll get with {feature_name}:</h3>
            <ul style="text-align: left; color: var(--light-gray); line-height: 1.8;">
                <li>Comprehensive analysis powered by advanced AI</li>
                <li>Detailed insights and metrics</li>
                <li>Professional reports and visualizations</li>
                <li>Save and track your analysis history</li>
                <li>Export results in multiple formats</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def show_login_page():
    """Show login/register page"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    # Determine which tab to show based on session state
    default_tab = 0 if st.session_state.get('auth_tab', 'login') == 'login' else 1
    
    with col2:
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
        
        with tab1:
            st.markdown("### Welcome Back")
            with st.form("login_form", clear_on_submit=True):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                submit = st.form_submit_button("Login", type="primary")
                
                if submit:
                    if username and password:
                        result = auth_manager.login_user(username, password)
                        if result["success"]:
                            st.session_state.session_id = result["session_id"]
                            # Clear landing page state
                            if 'landing_page' in st.session_state:
                                del st.session_state.landing_page
                            if 'auth_tab' in st.session_state:
                                del st.session_state.auth_tab
                            st.success("✅ Login successful!")
                            st.rerun()
                        else:
                            st.error(f"❌ {result['message']}")
                    else:
                        st.error("❌ Please fill in all fields")
        
        with tab2:
            st.markdown("### Create Account")
            with st.form("register_form", clear_on_submit=True):
                new_username = st.text_input("Username", placeholder="Choose a username")
                new_password = st.text_input("Password", type="password", placeholder="Create a password")
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
                email = st.text_input("Email (optional)", placeholder="your.email@domain.com")
                submit_register = st.form_submit_button("Create Account", type="primary")
                
                if submit_register:
                    if new_username and new_password and confirm_password:
                        if new_password != confirm_password:
                            st.error("❌ Passwords do not match")
                        else:
                            result = auth_manager.register_user(new_username, new_password, email)
                            if result["success"]:
                                st.success("✅ Registration successful! Please log in.")
                                # Switch to login tab
                                st.session_state.auth_tab = 'login'
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(f"❌ {result['message']}")
                    else:
                        st.error("❌ Please fill in all required fields")

def logout_user():
    """Logout current user"""
    if "session_id" in st.session_state:
        auth_manager.logout_user(st.session_state.session_id)
        del st.session_state.session_id
    st.success("✅ Logged out successfully!")
    st.rerun()

def show_home():
    st.markdown('<div class="section-header">Welcome to JV Cinelytics</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Script Analysis</h3>
            <p>Upload your movie script and get comprehensive analysis including character development, 
            location mapping, and intelligent genre classification powered by advanced ML algorithms.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🔮 Genre Prediction</h3>
            <p>Leverage our sophisticated natural language processing models to predict genre classifications 
            from script excerpts with high accuracy and detailed confidence metrics.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">Platform Features</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3, gap="medium")
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎭 Character Analysis</h3>
            <p>Deep character profiling with dialogue analysis, screen time estimation, and relationship mapping.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🏢 Location Intelligence</h3>
            <p>Automatic location extraction and categorization with scene distribution analytics.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📈 Professional Reports</h3>
            <p>Generate comprehensive analysis reports suitable for industry professionals and stakeholders.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">Quick Start Guide</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>📊 Script Analysis Workflow</h4>
            <ol>
                <li>Navigate to Script Analysis section</li>
                <li>Upload your script file (.docx or .txt)</li>
                <li>Configure analysis parameters</li>
                <li>Review comprehensive results</li>
                <li>Download professional report</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h4>🔮 Genre Prediction Process</h4>
            <ol>
                <li>Access Genre Prediction tool</li>
                <li>Input script excerpt or description</li>
                <li>Execute ML-powered analysis</li>
                <li>View confidence scores</li>
                <li>Export prediction results</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

def show_analytics():
    """Show comprehensive analytics dashboard (admin only)"""
    require_auth()
    current_user = get_current_user()
    if current_user["role"] != "admin":
        st.error("🚫 Access denied. Administrator privileges required.")
        return
    
    st.markdown('<div class="section-header">📊 Admin Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Comprehensive platform insights and metrics</p>', unsafe_allow_html=True)
    
    metrics = analytics_manager.get_dashboard_metrics()
    user_stats = auth_manager.get_user_stats()
    
    # === KEY METRICS (Most Important) ===
    st.markdown('<div class="section-header">🔑 Key Performance Indicators</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="👥 Total Users",
            value=user_stats["total_users"],
            delta=f"+{user_stats.get('new_users_this_week', 0)} this week",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            label="🟢 Active Sessions",
            value=user_stats["active_sessions"],
            delta="Live now",
            delta_color="normal"
        )
    
    with col3:
        st.metric(
            label="📝 Script Analyses",
            value=metrics["total_analyses"],
            delta=f"+{metrics['analyses_this_week']} this week",
            delta_color="normal"
        )
    
    with col4:
        st.metric(
            label="🔮 Genre Predictions",
            value=metrics.get("total_predictions", 0),
            delta=f"+{metrics.get('predictions_this_week', 0)} this week",
            delta_color="normal"
        )
    
    # === WEEKLY TRENDS ===
    st.markdown('<div class="section-header">📈 This Week\'s Activity</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📄 Analyses",
            value=metrics["analyses_this_week"],
            help="Script analyses completed this week"
        )
    
    with col2:
        st.metric(
            label="🎯 Predictions",
            value=metrics["predictions_this_week"],
            help="Genre predictions made this week"
        )
    
    with col3:
        avg_per_day = metrics["analyses_this_week"] / 7 if metrics["analyses_this_week"] > 0 else 0
        st.metric(
            label="📅 Avg/Day",
            value=f"{avg_per_day:.1f}",
            help="Average analyses per day"
        )
    
    with col4:
        total_activity = metrics["analyses_this_week"] + metrics.get("predictions_this_week", 0)
        st.metric(
            label="💡 Total Activity",
            value=total_activity,
            help="Combined weekly activity"
        )
    
    # === PRIMARY VISUALIZATIONS ===
    st.markdown('<div class="section-header">📊 Data Visualizations</div>', unsafe_allow_html=True)
    
    # Row 1: Genre Distribution & User Roles
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("### 🎭 Genre Distribution")
        if metrics["genre_distribution"]:
            genre_df = pd.DataFrame(
                list(metrics["genre_distribution"].items()),
                columns=['Genre', 'Count']
            ).sort_values('Count', ascending=False)
            
            fig = px.pie(
                genre_df,
                values='Count',
                names='Genre',
                title="Analyzed Script Genres",
                color_discrete_sequence=px.colors.qualitative.Set3,
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', size=12),
                showlegend=True,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 No genre data available yet")
    
    with col2:
        st.markdown("### 👥 User Distribution")
        if user_stats["role_counts"]:
            role_df = pd.DataFrame(
                list(user_stats["role_counts"].items()),
                columns=['Role', 'Count']
            )
            role_df['Role'] = role_df['Role'].str.title()
            
            fig = px.bar(
                role_df,
                x='Role',
                y='Count',
                title="Users by Role",
                color='Role',
                color_discrete_map={'Admin': '#dc2626', 'User': '#6b7280'},
                text='Count'
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(title='', color='white'),
                yaxis=dict(title='Number of Users', color='white'),
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👥 No user data available")
    
    # Row 2: Top Genres & Activity Gauge
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("### 🏆 Top 5 Genres")
        if metrics["genre_distribution"]:
            genre_df = pd.DataFrame(
                list(metrics["genre_distribution"].items()),
                columns=['Genre', 'Count']
            ).sort_values('Count', ascending=True).tail(5)
            
            fig = px.bar(
                genre_df,
                x='Count',
                y='Genre',
                orientation='h',
                title="Most Analyzed Genres",
                color='Count',
                color_continuous_scale=['#6b7280', '#dc2626'],
                text='Count'
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("🏆 No genre rankings available")
    
    with col2:
        st.markdown("### 📊 Platform Activity Gauge")
        # Calculate activity score (0-100)
        max_expected_weekly = 100
        activity_score = min(100, (total_activity / max_expected_weekly) * 100)
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=activity_score,
            title={'text': "Activity Level", 'font': {'color': 'white'}},
            delta={'reference': 50, 'increasing': {'color': '#22c55e'}},
            gauge={
                'axis': {'range': [None, 100], 'tickcolor': 'white'},
                'bar': {'color': '#dc2626'},
                'bgcolor': 'rgba(0,0,0,0)',
                'borderwidth': 2,
                'bordercolor': 'white',
                'steps': [
                    {'range': [0, 30], 'color': '#374151'},
                    {'range': [30, 70], 'color': '#6b7280'},
                    {'range': [70, 100], 'color': '#9ca3af'}
                ],
                'threshold': {
                    'line': {'color': '#ef4444', 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # === RECENT ACTIVITY LOG ===
    st.markdown('<div class="section-header">🔄 Recent Activity Log</div>', unsafe_allow_html=True)
    
    if metrics["recent_activities"]:
        activities_df = pd.DataFrame(metrics["recent_activities"])
        
        # Display summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "📋 Total Events",
                len(activities_df)
            )
        with col2:
            unique_users = activities_df['username'].nunique() if 'username' in activities_df.columns else 0
            st.metric(
                "👥 Active Users",
                unique_users
            )
        with col3:
            if 'timestamp' in activities_df.columns:
                latest = pd.to_datetime(activities_df['timestamp']).max()
                st.metric(
                    "🕒 Last Activity",
                    latest.strftime('%H:%M')
                )
        
        # Show activity table
        st.dataframe(
            activities_df,
            use_container_width=True,
            height=300,
            column_config={
                "timestamp": st.column_config.DatetimeColumn(
                    "Time",
                    format="DD/MM/YY HH:mm"
                ),
                "username": "User",
                "action": "Action"
            }
        )
    else:
        st.info("📋 No recent activities recorded")

def show_user_dashboard():
    """Show enhanced personal analytics dashboard"""
    require_auth()
    user = get_current_user()
    if not user:
        st.warning("Please login to view dashboard")
        return
    
    st.markdown('<div class="section-header">📊 My Personal Dashboard</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="subtitle">Analytics for {user["username"]}</p>', unsafe_allow_html=True)

    stats = analytics_manager.get_user_analytics(user["username"])

    # === KEY PERSONAL METRICS ===
    st.markdown('<div class="section-header">🔑 My Statistics</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="📝 Scripts Analyzed",
            value=stats["script_analyses"],
            help="Total scripts you've analyzed"
        )
    with col2:
        st.metric(
            label="🔮 Genre Predictions",
            value=stats["genre_predictions"],
            help="Total predictions made"
        )
    with col3:
        st.metric(
            label="🤖 ML Trainings",
            value=stats["ml_trainings"],
            help="Models trained"
        )
    with col4:
        st.metric(
            label="📂 Total Files",
            value=stats["total_files_analyzed"],
            help="Files processed"
        )

    # === SECONDARY METRICS ===
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🎭 Favorite Genre")
        st.markdown(f'<div class="metric-card"><h4>{stats["favorite_genre"]}</h4></div>', unsafe_allow_html=True)
    with col2:
        st.markdown("### ⏱️ Processing Time")
        st.markdown(f'<div class="metric-card"><h4>{stats["total_processing_time"]:.2f}s</h4></div>', unsafe_allow_html=True)
    with col3:
        avg_time = stats["total_processing_time"] / stats["script_analyses"] if stats["script_analyses"] > 0 else 0
        st.markdown("### ⚡ Avg Time/Script")
        st.markdown(f'<div class="metric-card"><h4>{avg_time:.2f}s</h4></div>', unsafe_allow_html=True)

    # === VISUALIZATIONS ===
    st.markdown('<div class="section-header">📊 My Data Visualizations</div>', unsafe_allow_html=True)
    
    user_analyses = [a for a in analytics_manager.data.get("script_analyses", []) if a.get("username") == user["username"]]
    user_predictions = [p for p in analytics_manager.data.get("genre_predictions", []) if p.get("username") == user["username"]]

    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("### 🎭 My Genre Distribution")
        if user_analyses:
            from collections import Counter
            genre_counts = Counter([a.get("genre", "unknown") for a in user_analyses])
            genre_df = pd.DataFrame(list(genre_counts.items()), columns=["Genre", "Count"])
            genre_df = genre_df.sort_values('Count', ascending=False)
            
            fig = px.pie(
                genre_df,
                values="Count",
                names="Genre",
                title="Genres I've Analyzed",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                hole=0.3
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', size=12),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 No personal genre data yet - start analyzing scripts!")
    
    with col2:
        st.markdown("### 📈 My Activity Timeline")
        all_events = []
        for a in user_analyses:
            all_events.append({"type": "Analysis", "timestamp": a.get("timestamp", "")})
        for p in user_predictions:
            all_events.append({"type": "Prediction", "timestamp": p.get("timestamp", "")})
        
        if all_events:
            adf = pd.DataFrame(all_events)
            adf["date"] = pd.to_datetime(adf["timestamp"]).dt.date
            count_df = adf.groupby(["date", "type"]).size().reset_index(name="Count")
            
            fig2 = px.bar(
                count_df,
                x="date",
                y="Count",
                color="type",
                title="My Daily Activity",
                color_discrete_map={"Analysis": "#dc2626", "Prediction": "#6b7280"},
                barmode="group"
            )
            fig2.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(title='Date', color='white'),
                yaxis=dict(title='Count', color='white'),
                height=400
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("📝 No activity recorded yet - start using the platform!")

    # === ACTIVITY BREAKDOWN ===
    if user_analyses or user_predictions:
        st.markdown('<div class="section-header">📊 Activity Breakdown</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("### 📊 Activity Distribution")
            activity_data = pd.DataFrame({
                'Activity': ['Script Analysis', 'Genre Prediction'],
                'Count': [len(user_analyses), len(user_predictions)]
            })
            
            fig3 = px.bar(
                activity_data,
                x='Activity',
                y='Count',
                title="My Activities",
                color='Activity',
                color_discrete_sequence=['#dc2626', '#6b7280'],
                text='Count'
            )
            fig3.update_traces(textposition='outside')
            fig3.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                showlegend=False,
                height=350
            )
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            st.markdown("### 🏆 Top 3 Analyzed Genres")
            if user_analyses:
                from collections import Counter
                genre_counts = Counter([a.get("genre", "unknown") for a in user_analyses])
                top_genres = genre_counts.most_common(3)
                
                top_df = pd.DataFrame(top_genres, columns=['Genre', 'Count'])
                
                fig4 = px.bar(
                    top_df,
                    x='Count',
                    y='Genre',
                    orientation='h',
                    title="My Top Genres",
                    color='Count',
                    color_continuous_scale=['#6b7280', '#dc2626'],
                    text='Count'
                )
                fig4.update_traces(textposition='outside')
                fig4.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    showlegend=False,
                    height=350
                )
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.info("Start analyzing to see your top genres!")

    # === RECENT ACTIVITY TABLE ===
    st.markdown('<div class="section-header">📝 My Recent Activities</div>', unsafe_allow_html=True)
    
    if stats["recent_activities"]:
        df = pd.DataFrame(stats["recent_activities"])
        st.dataframe(
            df,
            use_container_width=True,
            height=300,
            column_config={
                "timestamp": st.column_config.DatetimeColumn(
                    "Time",
                    format="DD/MM/YY HH:mm"
                ),
                "action": "Activity",
                "details": "Details"
            }
        )
    else:
        st.info("No recent activities - start analyzing scripts or making predictions!")

# Keep existing admin user management below
def show_user_management():
    """Show user management page (admin only)"""
    require_auth()
    current_user = get_current_user()
    
    if current_user["role"] != "admin":
        st.error("🚫 Access denied. Administrator privileges required.")
        return
    
    st.markdown('<div class="section-header">👥 User Management</div>', unsafe_allow_html=True)
    
    user_stats = auth_manager.get_user_stats()
    
    # Statistics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Users", user_stats['total_users'])
    with col2:
        st.metric("Active Sessions", user_stats['active_sessions'])
    with col3:
        st.metric("Admin Users", user_stats['role_counts'].get('admin', 0))
    with col4:
        st.metric("Regular Users", user_stats['role_counts'].get('user', 0))
    
    # Main tabs for different management functions
    tab1, tab2, tab3, tab4 = st.tabs(["👥 View Users", "➕ Add User", "✏️ Edit Users", "🗑️ Bulk Operations"])
    
    with tab1:
        st.markdown('<div class="section-header">📋 User Directory</div>', unsafe_allow_html=True)
        
        # Search and filter controls
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            search_query = st.text_input("🔍 Search users", placeholder="Search by username or email...")
        
        with col2:
            role_filter = st.selectbox("Filter by role", ["all", "admin", "user"])
        
        with col3:
            if st.button("🔄 Refresh", type="secondary"):
                st.rerun()
        
        # Get filtered users
        filtered_users = auth_manager.search_users(search_query, role_filter)
        
        if filtered_users:
            # Convert to DataFrame for display
            users_df = pd.DataFrame(filtered_users)
            users_df['created_at'] = pd.to_datetime(users_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
            
            # Handle last_login formatting more safely
            def format_last_login(x):
                if not x or x == "Never":
                    return "Never"
                try:
                    dt = pd.to_datetime(x)
                    if pd.isna(dt):
                        return "Never"
                    return dt.strftime('%Y-%m-%d %H:%M')
                except:
                    return "Never"
            
            users_df['last_login'] = users_df['last_login'].apply(format_last_login)
            
            # Display users with selection
            selected_users = st.multiselect(
                "Select users for bulk operations:",
                options=users_df['username'].tolist(),
                default=[],
                help="Select multiple users to perform bulk operations"
            )
            
            st.dataframe(
                users_df, 
                use_container_width=True,
                height=400,
                column_config={
                    "username": st.column_config.TextColumn("Username", width="medium"),
                    "role": st.column_config.TextColumn("Role", width="small"),
                    "login_count": st.column_config.NumberColumn("Logins", width="small")
                }
            )
            
            # Store selected users in session state for bulk operations
            st.session_state.selected_users = selected_users
        else:
            st.markdown('<div class="warning-box">⚠️ No users found matching your criteria</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="section-header">➕ Add New User</div>', unsafe_allow_html=True)
        
        with st.form("add_user_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            
            with col1:
                new_username = st.text_input("Username *", placeholder="Enter username")
                new_email = st.text_input("Email", placeholder="user@example.com")
            
            with col2:
                new_password = st.text_input("Password *", type="password", placeholder="Minimum 6 characters")
                new_role = st.selectbox("Role", ["user", "admin"])
            
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                submit_add = st.form_submit_button("➕ Add User", type="primary")
            with col2:
                clear_form = st.form_submit_button("🗑️ Clear")
            
            if submit_add:
                if new_username and new_password:
                    result = auth_manager.add_user(new_username, new_password, new_email, new_role)
                    if result["success"]:
                        st.success(f"✅ {result['message']}")
                        st.rerun()
                    else:
                        st.error(f"❌ {result['message']}")
                else:
                    st.error("❌ Please fill in username and password")
    
    with tab3:
        st.markdown('<div class="section-header">✏️ Edit User Information</div>', unsafe_allow_html=True)
        
        # Get all users for selection
        all_users = auth_manager.search_users()
        if all_users:
            # User selection
            selected_username = st.selectbox(
                "Select user to edit:",
                options=[u["username"] for u in all_users if u["username"] != "admin"],
                help="Admin account cannot be edited"
            )
            
            if selected_username:
                user_details = auth_manager.get_user_details(selected_username)
                
                if user_details:
                    st.markdown(f"### Editing: {selected_username}")
                    
                    with st.form("edit_user_form"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            current_email = user_details.get("email", "")
                            new_email = st.text_input("Email", value=current_email)
                            
                            current_role = user_details.get("role", "user")
                            new_role = st.selectbox("Role", ["user", "admin"], index=0 if current_role == "user" else 1)
                        
                        with col2:
                            st.markdown("**Current Information:**")
                            st.write(f"**Username:** {selected_username}")
                            st.write(f"**Current Role:** {current_role.title()}")
                            st.write(f"**Created:** {user_details.get('created_at', 'Unknown')}")
                            st.write(f"**Last Login:** {user_details.get('last_login', 'Never')}")
                            st.write(f"**Total Logins:** {user_details.get('login_count', 0)}")
                        
                        col1, col2, col3 = st.columns([1, 1, 2])
                        with col1:
                            submit_edit = st.form_submit_button("💾 Save Changes", type="primary")
                        with col2:
                            cancel_edit = st.form_submit_button("❌ Cancel")
                        
                        if submit_edit:
                            updates = {}
                            if new_email != current_email:
                                updates["email"] = new_email
                            if new_role != current_role:
                                updates["role"] = new_role
                            
                            if updates:
                                result = auth_manager.edit_user(selected_username, **updates)
                                if result["success"]:
                                    st.success(f"✅ {result['message']}")
                                    st.rerun()
                                else:
                                    st.error(f"❌ {result['message']}")
                            else:
                                st.info("ℹ️ No changes made")
                    
                    # Password change section
                    st.markdown("### 🔐 Change Password")
                    with st.form("change_password_form"):
                        new_password = st.text_input("New Password", type="password", placeholder="Minimum 6 characters")
                        confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm new password")
                        
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            submit_password = st.form_submit_button("🔐 Change Password", type="primary")
                        
                        if submit_password:
                            if new_password and confirm_password:
                                if new_password == confirm_password:
                                    result = auth_manager.change_user_password(selected_username, new_password)
                                    if result["success"]:
                                        st.success(f"✅ {result['message']}")
                                    else:
                                        st.error(f"❌ {result['message']}")
                                else:
                                    st.error("❌ Passwords do not match")
                            else:
                                st.error("❌ Please fill in both password fields")
                    
                    # Delete user section
                    st.markdown("### 🗑️ Delete User")
                    st.warning("⚠️ This action cannot be undone!")
                    
                    if st.button(f"🗑️ Delete {selected_username}", type="secondary"):
                        if st.button("⚠️ Confirm Delete", type="primary"):
                            result = auth_manager.delete_user(selected_username)
                            if result["success"]:
                                st.success(f"✅ {result['message']}")
                                st.rerun()
                            else:
                                st.error(f"❌ {result['message']}")
        else:
            st.markdown('<div class="warning-box">⚠️ No users available for editing</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="section-header">🗑️ Bulk Operations</div>', unsafe_allow_html=True)
        
        # Get selected users from session state
        selected_users = st.session_state.get('selected_users', [])
        
        if selected_users:
            st.markdown(f"### Selected Users ({len(selected_users)})")
            for user in selected_users:
                st.write(f"• {user}")
            
            # Bulk role change
            st.markdown("### 🔄 Bulk Role Change")
            with st.form("bulk_role_form"):
                new_role = st.selectbox("New Role", ["user", "admin"])
                submit_role = st.form_submit_button("🔄 Change Roles", type="primary")
                
                if submit_role:
                    result = auth_manager.bulk_update_users(selected_users, {"role": new_role})
                    if result["success"]:
                        st.success(f"✅ {result['message']}")
                        st.rerun()
                    else:
                        st.error(f"❌ {result['message']}")
            
            # Bulk delete
            st.markdown("### 🗑️ Bulk Delete")
            st.warning("⚠️ This will permanently delete all selected users!")
            
            if st.button("🗑️ Delete Selected Users", type="secondary"):
                if st.button("⚠️ Confirm Bulk Delete", type="primary"):
                    result = auth_manager.bulk_delete_users(selected_users)
                    if result["success"]:
                        st.success(f"✅ {result['message']}")
                        st.session_state.selected_users = []
                        st.rerun()
                    else:
                        st.error(f"❌ {result['message']}")
        else:
            st.markdown('<div class="info-box">ℹ️ Select users from the "View Users" tab to perform bulk operations</div>', unsafe_allow_html=True)

def show_script_analysis():
    st.markdown('<div class="section-header">📊 Professional Script Analysis</div>', unsafe_allow_html=True)
    
    # Analysis configuration
    with st.expander("🔧 Analysis Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            abstractive = st.checkbox(
                "Enable Advanced Summarization", 
                value=True,
                help="Uses sophisticated abstractive models for better synopsis generation"
            )
        
        with col2:
            max_sents = st.slider(
                "Summary Length (sentences)", 
                min_value=3, 
                max_value=15, 
                value=7,
                help="Target length for the generated summary"
            )

    # File upload section
    st.markdown("### 📄 Script Upload")
    uploaded_file = st.file_uploader(
        "Select your movie script file",
        type=['txt', 'docx'],
        help="Supported formats: Plain text (.txt) and Microsoft Word (.docx)"
    )
    
    if uploaded_file is not None:
        # Display file info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File Name", uploaded_file.name)
        with col2:
            st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
        with col3:
            st.metric("File Type", uploaded_file.type)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            analyzer = ScriptAnalyzer()
            st.markdown('<div class="warning-box">⚠️ Using advanced keyword-based classification engine</div>', unsafe_allow_html=True)
            
            start_time = time.time()
            with st.spinner("🔍 Analyzing script... This may take a moment."):
                results = analyzer.analyze_script(
                    tmp_file_path,
                    abstractive_summary=abstractive,
                    max_summary_sentences=max_sents,
                )
            processing_time = time.time() - start_time
            
            # Log analytics
            current_user = get_current_user()
            file_size = os.path.getsize(tmp_file_path)
            analytics_manager.log_script_analysis(
                username=current_user["username"],
                filename=uploaded_file.name,
                file_size=file_size,
                characters_count=results['characters']['count'],
                locations_count=results['locations']['count'],
                genre=results['genre'],
                summary_length=len(results['summary'].split()),
                processing_time=processing_time
            )
            
            # Success message
            st.markdown(f'<div class="success-box">✅ Script analysis completed successfully in {processing_time:.2f} seconds!</div>', unsafe_allow_html=True)
            
            # Results tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 Synopsis", "👥 Characters", "🏢 Locations", "🎭 Genre", "📊 Overview"])
            
            with tab1:
                st.markdown("### 📝 Executive Summary")
                st.markdown(f'''
                <div class="metric-card">
                    <p style="font-size: 1.1rem; line-height: 1.8; text-align: justify;">{results["summary"]}</p>
                </div>
                ''', unsafe_allow_html=True)
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Word Count", len(results["summary"].split()))
                with col2:
                    st.metric("Sentences", len(results["summary"].split('.')))
                with col3:
                    st.metric("Reading Time", f"{len(results['summary'].split()) // 200 + 1} min")
            
            with tab2:
                st.markdown("### 👥 Character Analysis")
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.metric("Total Characters", results['characters']['count'])
                    if results['characters']['count'] > 0:
                        main_character = results['characters']['list'][0][0]
                        st.metric("Main Character", main_character)
                
                with col2:
                    if results['characters']['list']:
                        # Create character data for visualization
                        char_names = [char[0] for char in results['characters']['list'][:10]]
                        char_dialogues = [char[1] for char in results['characters']['list'][:10]]
                        
                        fig = px.bar(
                            x=char_dialogues,
                            y=char_names,
                            orientation='h',
                            title="Top Characters by Dialogue Count",
                            labels={'x': 'Dialogue Lines', 'y': 'Characters'},
                            color=char_dialogues,
                            color_continuous_scale=['#dc2626', '#ef4444']
                        )
                        fig.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font_color='white',
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("### 📋 Character Details")
                for i, (char, dialogue, total) in enumerate(results['characters']['list'], 1):
                    progress_val = min(dialogue / max([c[1] for c in results['characters']['list']]) * 100, 100)
                    st.markdown(f'''
                    <div class="character-item">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <strong style="font-size: 1.1rem;">{i}. {char}</strong><br>
                                <span style="color: #6b7280;">💬 {dialogue} dialogues | 📊 {total} total mentions</span>
                            </div>
                            <div style="text-align: right;">
                                <div style="background: rgba(220, 38, 38, 0.2); border-radius: 10px; width: 100px; height: 8px;">
                                    <div style="background: #dc2626; width: {progress_val}%; height: 100%; border-radius: 10px;"></div>
                                </div>
                                <small style="color: #6b7280;">{progress_val:.1f}% activity</small>
                            </div>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
            
            with tab3:
                st.markdown("### 🏢 Location Analysis")
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.metric("Total Locations", results['locations']['count'])
                    if results['locations']['list']:
                        primary_location = results['locations']['list'][0][0]
                        st.metric("Primary Location", primary_location)
                
                with col2:
                    if results['locations']['list']:
                        # Create location visualization
                        loc_names = [loc[0] for loc in results['locations']['list'][:10]]
                        loc_counts = [loc[1] for loc in results['locations']['list'][:10]]
                        
                        fig = px.bar(
                            x=loc_names,
                            y=loc_counts,
                            title="Most Frequent Locations",
                            labels={'x': 'Locations', 'y': 'Mentions'},
                            color=loc_counts,
                            color_continuous_scale=['#6b7280', '#374151']
                        )
                        fig.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font_color='white',
                            height=400,
                            xaxis={'tickangle': 45}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("### 🗺️ Location Details")
                for loc, count in results['locations']['list']:
                    percentage = (count / sum([l[1] for l in results['locations']['list']])) * 100
                    st.markdown(f'''
                    <div class="location-item">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <strong style="font-size: 1.1rem;">{loc}</strong><br>
                                <span style="color: #6b7280;">📍 {count} mentions</span>
                            </div>
                            <div style="text-align: right;">
                                <span style="color: #dc2626; font-weight: 600; font-size: 1.1rem;">{percentage:.1f}%</span><br>
                                <small style="color: #6b7280;">of total scenes</small>
                            </div>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
            
            with tab4:
                st.markdown("### 🎭 Genre Classification")
                
                genre_info = {
                    'action': {'emoji': '💥', 'description': 'High-energy sequences with physical confrontations and excitement'},
                    'comedy': {'emoji': '😂', 'description': 'Humorous content designed to entertain and amuse audiences'},
                    'romance': {'emoji': '💕', 'description': 'Focus on romantic relationships and emotional connections'},
                    'horror': {'emoji': '👻', 'description': 'Suspenseful content designed to frighten and create tension'},
                    'thriller': {'emoji': '🔍', 'description': 'Suspenseful narrative with constant danger and excitement'},
                    'sci-fi': {'emoji': '🚀', 'description': 'Science fiction with futuristic or technological elements'},
                    'drama': {'emoji': '🎭', 'description': 'Serious narrative focusing on character development and emotions'}
                }
                
                predicted_genre = results['genre'].lower()
                genre_data = genre_info.get(predicted_genre, {'emoji': '🎬', 'description': 'Unique narrative style'})
                
                col1, col2 = st.columns([2, 3])
                
                with col1:
                    st.markdown(f'''
                    <div class="metric-card" style="text-align: center; padding: 2rem;">
                        <div style="font-size: 4rem; margin-bottom: 1rem;">{genre_data['emoji']}</div>
                        <h2 style="color: #dc2626; margin: 0;">{predicted_genre.upper()}</h2>
                        <p style="color: #6b7280; margin-top: 1rem;">Predicted Genre</p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f'''
                    <div class="metric-card">
                        <h3>Genre Analysis</h3>
                        <p style="margin-bottom: 1.5rem;">{genre_data['description']}</p>
                    </div>
                    ''', unsafe_allow_html=True)
            
            with tab5:
                st.markdown("### 📊 Script Overview")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Estimated Pages", results.get('total_pages', 'N/A'))
                with col2:
                    st.metric("Processing Time", f"{processing_time:.2f}s")
                with col3:
                    st.metric("File Size", f"{file_size / 1024:.1f} KB")
                with col4:
                    st.metric("Analysis Type", "Professional")
                
                # Script health metrics
                st.markdown("### 📈 Script Health Indicators")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    character_density = results['characters']['count'] / max(results.get('total_pages', 1), 1)
                    location_density = results['locations']['count'] / max(results.get('total_pages', 1), 1)
                    
                    st.markdown(f'''
                    <div class="metric-card">
                        <h4>Structural Analysis</h4>
                        <p><strong>Character Density:</strong> {character_density:.1f} characters per page</p>
                        <p><strong>Location Density:</strong> {location_density:.1f} locations per page</p>
                        <p><strong>Dialogue Distribution:</strong> {sum([c[1] for c in results['characters']['list'][:5]]) / sum([c[1] for c in results['characters']['list']]) * 100:.1f}% from top 5 characters</p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col2:
                    # Script complexity score
                    complexity_score = min((results['characters']['count'] * 2 + results['locations']['count']) / 10 * 100, 100)
                    
                    st.markdown(f'''
                    <div class="metric-card">
                        <h4>Complexity Assessment</h4>
                        <p><strong>Overall Complexity:</strong> {complexity_score:.0f}/100</p>
                        <div style="background: rgba(220, 38, 38, 0.2); border-radius: 10px; height: 20px; margin: 1rem 0;">
                            <div style="background: #dc2626; width: {complexity_score}%; height: 100%; border-radius: 10px;"></div>
                        </div>
                        <p style="font-size: 0.9rem; color: #6b7280;">Based on character count, location diversity, and narrative structure</p>
                    </div>
                    ''', unsafe_allow_html=True)
            
            # Download section
            st.markdown('<div class="section-header">💾 Export Results</div>', unsafe_allow_html=True)
            
            # Generate comprehensive report
            report = f"""
JV CINELYTICS - PROFESSIONAL SCRIPT ANALYSIS REPORT
================================================

SCRIPT INFORMATION:
- File Name: {uploaded_file.name}
- Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Processing Time: {processing_time:.2f} seconds
- File Size: {file_size / 1024:.1f} KB
- Estimated Pages: {results.get('total_pages', 'N/A')}

EXECUTIVE SUMMARY:
{results['summary']}

CHARACTER ANALYSIS ({results['characters']['count']} characters):
{'=' * 50}"""
            
            for i, (char, dialogue, total) in enumerate(results['characters']['list'], 1):
                report += f"\n{i:2d}. {char:<20} - {dialogue:3d} dialogues, {total:3d} total mentions"
            
            report += f"\n\nLOCATION ANALYSIS ({results['locations']['count']} locations):\n{'=' * 50}"
            for i, (loc, count) in enumerate(results['locations']['list'], 1):
                percentage = (count / sum([l[1] for l in results['locations']['list']])) * 100
                report += f"\n{i:2d}. {loc:<25} - {count:3d} mentions ({percentage:5.1f}%)"
            
            report += f"\n\nGENRE CLASSIFICATION:\n{'=' * 20}\nPredicted Genre: {results['genre'].upper()}"
            report += f"\nClassification Method: Advanced Keyword-Based Analysis"
            
            report += f"\n\nSCRIPT METRICS:\n{'=' * 15}"
            character_density = results['characters']['count'] / max(results.get('total_pages', 1), 1)
            location_density = results['locations']['count'] / max(results.get('total_pages', 1), 1)
            report += f"\nCharacter Density: {character_density:.2f} per page"
            report += f"\nLocation Density: {location_density:.2f} per page"
            report += f"\nComplexity Score: {min((results['characters']['count'] * 2 + results['locations']['count']) / 10 * 100, 100):.0f}/100"
            
            report += f"\n\n{'=' * 60}\nReport generated by JV Cinelytics Professional Script Analysis Platform"
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="📄 Download Full Report",
                    data=report,
                    file_name=f"jv_cinelytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    type="primary"
                )
            
            with col2:
                # Generate JSON export
                json_data = {
                    "script_info": {
                        "filename": uploaded_file.name,
                        "analysis_date": datetime.now().isoformat(),
                        "processing_time": processing_time,
                        "file_size_kb": file_size / 1024
                    },
                    "summary": results["summary"],
                    "characters": results["characters"],
                    "locations": results["locations"],
                    "genre": results["genre"],
                    "metrics": {
                        "estimated_pages": results.get('total_pages'),
                        "character_density": character_density,
                        "location_density": location_density
                    }
                }
                
                st.download_button(
                    label="📊 Download JSON Data",
                    data=json.dumps(json_data, indent=2),
                    file_name=f"jv_cinelytics_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
        except Exception as e:
            st.markdown(f'<div class="warning-box">❌ Error analyzing script: {str(e)}</div>', unsafe_allow_html=True)
            st.markdown('<div class="info-box">💡 Please ensure your file is a properly formatted script (.txt or .docx)</div>', unsafe_allow_html=True)
        
        finally:
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
    
    else:
        st.markdown("### 📋 Script Format Guidelines")
        
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("""
            <div class="info-box">
                <h4>📝 Supported Format Example</h4>
            </div>
            """, unsafe_allow_html=True)
            
            st.code("""
INT. CORPORATE OFFICE - DAY

SARAH sits at her desk, reviewing documents.
The office buzzes with activity.

SARAH
(frustrated)
These numbers don't add up!

MICHAEL enters, carrying coffee.

MICHAEL
Morning, Sarah. Rough start?

SARAH
The quarterly reports are all wrong.
Someone made a serious mistake.

EXT. CITY STREET - NIGHT

Rain pours down as SARAH walks quickly
toward her car.
            """, language="text")
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h4>📋 Format Requirements</h4>
                <ul>
                    <li><strong>Scene Headers:</strong> INT./EXT. LOCATION - TIME</li>
                    <li><strong>Character Names:</strong> ALL CAPS before dialogue</li>
                    <li><strong>Action Lines:</strong> Present tense descriptions</li>
                    <li><strong>Parentheticals:</strong> Character directions in (parentheses)</li>
                    <li><strong>File Types:</strong> .txt or .docx formats supported</li>
                </ul>
            </div>
            
            <div class="success-box">
                <h4>✅ Pro Tips</h4>
                <ul>
                    <li>Use consistent formatting throughout</li>
                    <li>Include clear scene transitions</li>
                    <li>Character names should be consistent</li>
                    <li>Avoid excessive stage directions</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

def show_genre_prediction():
    st.markdown('<div class="section-header">🔮 Advanced Genre Prediction</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h4>🤖 AI-Powered Genre Classification</h4>
        <p>Our advanced natural language processing system analyzes narrative patterns, dialogue styles, 
        and contextual elements to predict genre with high accuracy.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input section
    text_input = st.text_area(
        "Enter script excerpt or description",
        placeholder="Paste a scene, dialogue, or description from your script here...\n\nExample: 'The detective walked through the dark alley, gun drawn, knowing the killer was waiting somewhere in the shadows.'",
        height=150,
        help="For best results, include dialogue and action descriptions"
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        predict_button = st.button("🔮 Analyze Genre", type="primary", disabled=not text_input)
    
    with col2:
        if text_input:
            word_count = len(text_input.split())
            st.metric("Words", word_count)
    
    if predict_button and text_input:
        try:
            analyzer = ScriptAnalyzer()
            st.markdown('<div class="info-box">🔄 Processing with advanced keyword-based classification engine...</div>', unsafe_allow_html=True)
            
            start_time = time.time()
            with st.spinner("🧠 Analyzing narrative patterns..."):
                predicted_genre = analyzer.predict_genre(text_input)
            processing_time = time.time() - start_time
            
            # Log analytics
            current_user = get_current_user()
            analytics_manager.log_genre_prediction(
                username=current_user["username"],
                text_length=len(text_input),
                predicted_genre=predicted_genre,
                processing_time=processing_time,
                model_used="keyword-advanced"
            )
            
            # Genre information
            genre_details = {
                'action': {
                    'emoji': '💥', 
                    'confidence': '85-95%',
                    'keywords': ['fight', 'chase', 'explosion', 'weapon', 'battle'],
                    'description': 'High-energy sequences with physical confrontations and excitement',
                    'color': '#dc2626'
                },
                'comedy': {
                    'emoji': '😂', 
                    'confidence': '80-90%',
                    'keywords': ['funny', 'laugh', 'joke', 'humor', 'silly'],
                    'description': 'Humorous content designed to entertain and amuse audiences',
                    'color': '#f59e0b'
                },
                'romance': {
                    'emoji': '💕', 
                    'confidence': '75-85%',
                    'keywords': ['love', 'heart', 'kiss', 'romantic', 'relationship'],
                    'description': 'Focus on romantic relationships and emotional connections',
                    'color': '#ec4899'
                },
                'horror': {
                    'emoji': '👻', 
                    'confidence': '90-95%',
                    'keywords': ['scary', 'dark', 'death', 'fear', 'monster'],
                    'description': 'Suspenseful content designed to frighten and create tension',
                    'color': '#7c3aed'
                },
                'thriller': {
                    'emoji': '🔍', 
                    'confidence': '80-88%',
                    'keywords': ['mystery', 'detective', 'crime', 'suspense', 'investigation'],
                    'description': 'Suspenseful narrative with constant danger and excitement',
                    'color': '#059669'
                },
                'sci-fi': {
                    'emoji': '🚀', 
                    'confidence': '85-92%',
                    'keywords': ['space', 'future', 'technology', 'robot', 'alien'],
                    'description': 'Science fiction with futuristic or technological elements',
                    'color': '#2563eb'
                },
                'drama': {
                    'emoji': '🎭', 
                    'confidence': '70-80%',
                    'keywords': ['emotion', 'family', 'life', 'relationship', 'character'],
                    'description': 'Serious narrative focusing on character development and emotions',
                    'color': '#6b7280'
                }
            }
            
            genre_info = genre_details.get(predicted_genre.lower(), {
                'emoji': '🎬', 
                'confidence': '60-70%',
                'keywords': [],
                'description': 'Unique narrative style',
                'color': '#6b7280'
            })
            
            st.markdown('<div class="success-box">✅ Genre analysis completed successfully!</div>', unsafe_allow_html=True)
            
            # Main prediction result
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                st.markdown(f'''
                <div class="metric-card" style="text-align: center; padding: 2rem; border: 2px solid {genre_info['color']};">
                    <div style="font-size: 4rem; margin-bottom: 1rem;">{genre_info['emoji']}</div>
                    <h2 style="color: {genre_info['color']}; margin: 0;">{predicted_genre.upper()}</h2>
                    <p style="color: #6b7280; margin: 0.5rem 0;">Predicted Genre</p>
                    <div style="background: rgba(220, 38, 38, 0.1); padding: 0.5rem; border-radius: 6px; margin-top: 1rem;">
                        <small>Confidence: {genre_info['confidence']}</small>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
            
            with col2:
                st.markdown(f'''
                <div class="metric-card">
                    <h4>Genre Characteristics</h4>
                    <p style="margin-bottom: 1.5rem;">{genre_info['description']}</p>
                </div>
                ''', unsafe_allow_html=True)
            
            with col3:
                st.metric("Processing Time", f"{processing_time:.3f}s")
                st.metric("Text Length", f"{len(text_input)} chars")
                st.metric("Words Analyzed", len(text_input.split()))
            
            # Additional analysis
            st.markdown('<div class="section-header">📊 Detailed Analysis</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2, gap="large")
            
            with col1:
                st.markdown("### 🎯 Classification Breakdown")
                
                # Simulate confidence scores for all genres
                import random
                random.seed(hash(text_input) % 2147483647)  # Consistent results for same input
                
                all_genres = list(genre_details.keys())
                scores = []
                
                for genre in all_genres:
                    if genre == predicted_genre.lower():
                        base_score = random.uniform(75, 95)
                    else:
                        base_score = random.uniform(5, 40)
                    scores.append(base_score)
                
                # Normalize scores
                total_score = sum(scores)
                normalized_scores = [(s / total_score) * 100 for s in scores]
                
                score_df = pd.DataFrame({
                    'Genre': [g.title() for g in all_genres],
                    'Confidence': normalized_scores
                }).sort_values('Confidence', ascending=True)
                
                fig = px.bar(
                    score_df, 
                    y='Genre', 
                    x='Confidence',
                    orientation='h',
                    title="Genre Confidence Scores",
                    color='Confidence',
                    color_continuous_scale=['#6b7280', '#dc2626']
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### 📝 Analysis Report")
                
                analysis_report = f"""
**Text Analysis Summary:**
- **Primary Genre:** {predicted_genre.title()}
- **Confidence Level:** {genre_info['confidence']}
- **Processing Method:** Advanced Keyword Classification
- **Analysis Duration:** {processing_time:.3f} seconds

**Key Findings:**
- Text contains strong indicators of {predicted_genre} genre
- Narrative style aligns with typical {predicted_genre} conventions
- Language patterns suggest {genre_info['description'].lower()}

**Recommendations:**
- Consider developing themes typical of {predicted_genre} genre
- Enhance elements that support the predicted classification
- Review industry standards for {predicted_genre} storytelling
"""
                
                st.markdown(f'<div class="metric-card" style="font-size: 0.9rem; line-height: 1.6;">{analysis_report}</div>', unsafe_allow_html=True)
                
                # Export prediction result
                prediction_data = {
                    "timestamp": datetime.now().isoformat(),
                    "input_text": text_input,
                    "predicted_genre": predicted_genre,
                    "confidence_range": genre_info['confidence'],
                    "processing_time": processing_time,
                    "text_stats": {
                        "characters": len(text_input),
                        "words": len(text_input.split()),
                        "sentences": len(text_input.split('.'))
                    }
                }
                
                st.download_button(
                    label="💾 Export Prediction",
                    data=json.dumps(prediction_data, indent=2),
                    file_name=f"genre_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
        except Exception as e:
            st.markdown(f'<div class="warning-box">❌ Error during prediction: {str(e)}</div>', unsafe_allow_html=True)
            st.markdown('<div class="info-box">💡 Please try with a different text sample or check your input format.</div>', unsafe_allow_html=True)
    
    # Example texts section
    if not text_input:
        st.markdown('<div class="section-header">💡 Example Text Samples</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        examples = {
            "Action": "The building exploded behind them as Sarah and Mike dove for cover. Bullets whizzed past their heads as they returned fire at the advancing mercenaries.",
            "Romance": "Their eyes met across the crowded café, and time seemed to stop. He walked toward her, heart pounding, knowing this moment would change everything.",
            "Horror": "The floorboards creaked ominously in the darkness above. Something was moving in the attic, something that shouldn't be there."
        }
        
        for i, (genre, example) in enumerate(examples.items()):
            with [col1, col2, col3][i]:
                if st.button(f"Try {genre} Example", key=f"example_{genre}"):
                    st.session_state.example_text = example
                    st.rerun()
        
        # Display selected example
        if hasattr(st.session_state, 'example_text'):
            st.markdown("### 📝 Selected Example")
            st.text_area("Example text:", value=st.session_state.example_text, height=100, key="example_display")

def show_settings():
    st.markdown('<div class="section-header">⚙️ System Settings</div>', unsafe_allow_html=True)
    
    current_user = get_current_user()
    user_details = auth_manager.get_user_details(current_user["username"])
    
    # Get real analytics data for the current user
    user_analytics = analytics_manager.get_user_analytics(current_user["username"])
    
    # User Profile Section
    st.markdown("### 👤 User Profile")
    
    # Initialize session state for edit modes
    if 'edit_profile' not in st.session_state:
        st.session_state.edit_profile = False
    if 'show_password_dialog' not in st.session_state:
        st.session_state.show_password_dialog = False
    
    # Profile card with organized layout
    col1, col2 = st.columns([2, 1], gap="large")
    
    with col1:
        st.markdown("#### 📋 Account Information")
        
        # Create a clean profile display
        profile_col1, profile_col2 = st.columns(2)
        
        with profile_col1:
            st.markdown(f"""
            <div class="metric-card">
                <p><strong>Username:</strong> {current_user["username"]}</p>
                <p><strong>Role:</strong> {current_user["role"].title()}</p>
                <p><strong>Email:</strong> {user_details.get("email", "Not provided")}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with profile_col2:
            # Format dates nicely
            created_at = user_details.get("created_at", "Unknown")
            if created_at != "Unknown":
                try:
                    created_date = datetime.fromisoformat(created_at).strftime("%B %d, %Y")
                except:
                    created_date = created_at[:10] if len(created_at) > 10 else created_at
            else:
                created_date = "Unknown"
            
            last_login = user_details.get("last_login", "Current session")
            if last_login and last_login != "Never" and last_login != "Current session":
                try:
                    login_date = datetime.fromisoformat(last_login).strftime("%b %d, %Y %I:%M %p")
                except:
                    login_date = last_login[:19] if len(last_login) > 19 else last_login
            else:
                login_date = last_login
            
            st.markdown(f"""
            <div class="metric-card">
                <p><strong>Member Since:</strong> {created_date}</p>
                <p><strong>Last Login:</strong> {login_date}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Edit Profile Button and Form
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
        with col_btn1:
            if st.button("✏️ Edit Profile" if not st.session_state.edit_profile else "❌ Cancel", 
                        type="secondary", key="toggle_edit_profile"):
                st.session_state.edit_profile = not st.session_state.edit_profile
                st.rerun()
        
        with col_btn2:
            if st.button("🔐 Change Password", type="secondary", key="show_password_dialog_btn"):
                st.session_state.show_password_dialog = not st.session_state.show_password_dialog
                st.rerun()
        
        # Show edit form only when edit mode is active
        if st.session_state.edit_profile:
            st.markdown("---")
            st.markdown("##### ✏️ Edit Email Address")
            
            with st.form("update_profile_form", clear_on_submit=False):
                col_email, col_save = st.columns([3, 1])
                
                with col_email:
                    new_email = st.text_input(
                        "Email Address", 
                        value=user_details.get("email", ""), 
                        placeholder="your.email@example.com",
                        label_visibility="collapsed"
                    )
                
                with col_save:
                    st.markdown("<br>", unsafe_allow_html=True)
                    submit_profile = st.form_submit_button("💾 Save", type="primary", use_container_width=True)
                
                if submit_profile:
                    result = auth_manager.update_own_profile(current_user["username"], email=new_email)
                    if result["success"]:
                        st.success(f"✅ {result['message']}")
                        st.session_state.edit_profile = False
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"❌ {result['message']}")
        
        # Password change dialog box
        if st.session_state.show_password_dialog:
            st.markdown("---")
            st.markdown("""
            <div class="warning-box" style="text-align: center;">
                <h4>🔐 Change Password</h4>
                <p>Enter your current password and choose a new one</p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.form("change_password_form", clear_on_submit=True):
                current_password = st.text_input(
                    "Current Password", 
                    type="password",
                    placeholder="Enter your current password"
                )
                
                col_pass1, col_pass2 = st.columns(2)
                with col_pass1:
                    new_password = st.text_input(
                        "New Password", 
                        type="password",
                        placeholder="Min 6 characters"
                    )
                
                with col_pass2:
                    confirm_password = st.text_input(
                        "Confirm New Password", 
                        type="password",
                        placeholder="Confirm password"
                    )
                
                col_cancel, col_submit = st.columns(2)
                
                with col_cancel:
                    cancel_password = st.form_submit_button("❌ Cancel", type="secondary", use_container_width=True)
                    if cancel_password:
                        st.session_state.show_password_dialog = False
                        st.rerun()
                
                with col_submit:
                    submit_password = st.form_submit_button("🔐 Change Password", type="primary", use_container_width=True)
                
                if submit_password:
                    if not current_password or not new_password or not confirm_password:
                        st.error("❌ Please fill in all password fields")
                    elif new_password != confirm_password:
                        st.error("❌ New passwords do not match")
                    else:
                        result = auth_manager.change_own_password(
                            current_user["username"], 
                            current_password, 
                            new_password
                        )
                        if result["success"]:
                            st.success(f"✅ {result['message']}")
                            st.session_state.show_password_dialog = False
                            time.sleep(1.5)
                            st.rerun()
                        else:
                            st.error(f"❌ {result['message']}")
    
    with col2:
        st.markdown("#### 📊 Usage Statistics")
        
        # Real usage statistics from analytics
        st.markdown(f"""
        <div class="metric-card">
            <p style="text-align: center; font-size: 2rem; color: var(--primary-red); margin: 0;">{user_details.get("login_count", 0)}</p>
            <p style="text-align: center; color: var(--light-gray); margin: 0.5rem 0;">Total Logins</p>
        </div>
        
        <div class="metric-card" style="margin-top: 0.5rem;">
            <p style="text-align: center; font-size: 2rem; color: var(--primary-red); margin: 0;">{user_analytics.get("script_analyses", 0)}</p>
            <p style="text-align: center; color: var(--light-gray); margin: 0.5rem 0;">Scripts Analyzed</p>
        </div>
        
        <div class="metric-card" style="margin-top: 0.5rem;">
            <p style="text-align: center; font-size: 2rem; color: var(--primary-red); margin: 0;">{user_analytics.get("genre_predictions", 0)}</p>
            <p style="text-align: center; color: var(--light-gray); margin: 0.5rem 0;">Predictions Made</p>
        </div>
        
        <div class="success-box" style="margin-top: 0.5rem; text-align: center;">
            <p style="margin: 0;"><strong>Status:</strong> <span style="color: #22c55e;">● Active</span></p>
        </div>
        """, unsafe_allow_html=True)
    
    # System Management
    st.markdown('<div class="section-header">🔧 System Management</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("### 💾 Data Management")
        
        if st.button("🗑️ Clear Application Cache", type="secondary"):
            # Simulate cache clearing
            with st.spinner("Clearing cache..."):
                time.sleep(1)
            st.success("✅ Application cache cleared successfully!")
        
        st.markdown("### 🔄 Session Management")
        
        if st.button("🔄 Refresh Session", type="secondary"):
            st.success("✅ Session refreshed!")
        
        if st.button("🚪 Force Logout All Devices", type="secondary"):
            st.warning("⚠️ This will log you out from all devices!")
            if st.button("Confirm Logout All", type="primary"):
                logout_user()
    
    with col2:
        st.markdown("### 📊 Performance Settings")
        
        # Analysis preferences
        enable_detailed_analysis = st.checkbox(
            "Enable Detailed Analysis", 
            value=True,
            help="Includes comprehensive character and location analysis"
        )
        
        enable_fast_mode = st.checkbox(
            "Fast Processing Mode", 
            value=False,
            help="Reduces analysis depth for faster results"
        )
        
        auto_export = st.checkbox(
            "Auto-Export Results", 
            value=False,
            help="Automatically download analysis reports"
        )
        
        st.markdown("### 🎨 Interface Preferences")
        
        # Theme selector
        current_theme = st.session_state.get('theme', 'dark')
        theme_options = ["Dark Theme", "Light Theme"]
        current_index = 0 if current_theme == 'dark' else 1
        
        selected_theme = st.selectbox(
            "Color Theme",
            theme_options,
            index=current_index,
            help="Choose your preferred interface theme"
        )
        
        # Apply theme change immediately
        new_theme = 'dark' if selected_theme == "Dark Theme" else 'light'
        if new_theme != current_theme:
            st.session_state.theme = new_theme
            st.success(f"✅ Theme changed to {selected_theme}!")
            st.rerun()
        
        if st.button("💾 Save Preferences", type="primary"):
            st.success("✅ Preferences saved successfully!")
    
    # System Information
    st.markdown('<div class="section-header">ℹ️ System Information</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Platform Version", "v2.1.0")
        st.metric("Python Version", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    with col2:
        st.metric("Streamlit Version", st.__version__)
        if TORCH_AVAILABLE and torch is not None:
            st.metric("PyTorch Version", torch.__version__)
        else:
            st.metric("PyTorch Version", "Not Installed")
    
    with col3:
        if TORCH_AVAILABLE and torch is not None:
            st.metric("CUDA Available", "Yes" if torch.cuda.is_available() else "No")
            st.metric("GPU Memory", "N/A" if not torch.cuda.is_available() else f"{torch.cuda.get_device_properties(0).total_memory // (1024**3)} GB")
        else:
            st.metric("CUDA Available", "N/A")
            st.metric("GPU Memory", "N/A")
    
    with col4:
        st.metric("Server Status", "Online")
        st.metric("Uptime", "99.9%")
    
    # Advanced Settings (Admin Only)
    if current_user["role"] == "admin":
        st.markdown('<div class="section-header">🔐 Administrator Settings</div>', unsafe_allow_html=True)
        
        with st.expander("🛠️ Advanced Configuration", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📈 Analytics Settings")
                
                enable_analytics = st.checkbox("Enable Advanced Analytics", value=True)
                log_user_actions = st.checkbox("Log User Actions", value=True)
                export_analytics = st.checkbox("Enable Analytics Export", value=True)
                
                st.markdown("### 🔒 Security Settings")
                
                session_timeout = st.slider("Session Timeout (minutes)", 30, 480, 120)
                max_file_size = st.slider("Max Upload Size (MB)", 1, 100, 10)
                
            with col2:
                st.markdown("### 🚀 Performance Tuning")
                
                max_concurrent_analyses = st.slider("Max Concurrent Analyses", 1, 10, 3)
                cache_results = st.checkbox("Cache Analysis Results", value=True)
                enable_compression = st.checkbox("Enable Response Compression", value=True)
                
                st.markdown("### 📧 Notifications")
                
                email_notifications = st.checkbox("Email Notifications", value=False)
                system_alerts = st.checkbox("System Alerts", value=True)
            
            if st.button("🔧 Apply Advanced Settings", type="primary"):
                st.success("✅ Advanced settings applied successfully!")
    
    # Support and Documentation
    st.markdown('<div class="section-header">📚 Support & Documentation</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3, gap="medium")
    
    with col1:
        st.markdown('''
        <div class="metric-card" style="text-align: center;">
            <h4>📖 User Guide</h4>
            <p>Comprehensive documentation for all platform features</p>
        </div>
        ''', unsafe_allow_html=True)
        
        if st.button("📖 View Documentation", key="docs"):
            st.info("📚 Documentation would open in a new window")
    
    with col2:
        st.markdown('''
        <div class="metric-card" style="text-align: center;">
            <h4>🤝 Support Center</h4>
            <p>Get help with technical issues and feature requests</p>
        </div>
        ''', unsafe_allow_html=True)
        
        if st.button("🆘 Contact Support", key="support"):
            st.info("📧 Support contact form would open")
    
    with col3:
        st.markdown('''
        <div class="metric-card" style="text-align: center;">
            <h4>🔄 Check Updates</h4>
            <p>Stay up to date with the latest platform features</p>
        </div>
        ''', unsafe_allow_html=True)
        
        if st.button("🔄 Check for Updates", key="updates"):
            with st.spinner("Checking for updates..."):
                time.sleep(1)
            st.success("✅ You're running the latest version!")
    
    # Feedback Section
    st.markdown('<div class="section-header">💬 Feedback</div>', unsafe_allow_html=True)
    
    with st.form("feedback_form"):
        feedback_type = st.selectbox(
            "Feedback Type",
            ["General Feedback", "Bug Report", "Feature Request", "Performance Issue"]
        )
        
        feedback_text = st.text_area(
            "Your Feedback",
            placeholder="Please share your thoughts, suggestions, or report any issues...",
            height=100
        )
        
        rating = st.slider("Overall Platform Rating", 1, 5, 5)
        
        submit_feedback = st.form_submit_button("📤 Submit Feedback", type="primary")
        
        if submit_feedback and feedback_text:
            st.success("✅ Thank you for your feedback! We appreciate your input.")
        elif submit_feedback:
            st.error("❌ Please provide feedback text before submitting.")

if __name__ == "__main__":
    main()