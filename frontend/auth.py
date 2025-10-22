import streamlit as st
import requests
import json
from datetime import datetime, timedelta

# API Configuration
API_BASE_URL = "http://localhost:8000/api"

def init_session_state():
    """Initialize session state for authentication"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'access_token' not in st.session_state:
        st.session_state.access_token = None
    if 'refresh_token' not in st.session_state:
        st.session_state.refresh_token = None

def get_auth_headers():
    """Get headers with authentication token"""
    if st.session_state.access_token:
        return {
            'Authorization': f'Bearer {st.session_state.access_token}',
            'Content-Type': 'application/json'
        }
    return {'Content-Type': 'application/json'}

def signup_user(username, email, password):
    """Register a new user"""
    try:
        response = requests.post(f"{API_BASE_URL}/auth/signup/", json={
            'username': username,
            'email': email,
            'password': password
        })
        
        if response.status_code == 201:
            data = response.json()
            st.session_state.authenticated = True
            st.session_state.user = data['user']
            st.session_state.access_token = data['tokens']['access']
            st.session_state.refresh_token = data['tokens']['refresh']
            return True, "Registration successful!"
        else:
            error_data = response.json()
            return False, error_data.get('error', 'Registration failed')
    except Exception as e:
        return False, f"Connection error: {str(e)}"

def login_user(username, password):
    """Login user"""
    try:
        response = requests.post(f"{API_BASE_URL}/auth/login/", json={
            'username': username,
            'password': password
        })
        
        if response.status_code == 200:
            data = response.json()
            st.session_state.authenticated = True
            st.session_state.user = data['user']
            st.session_state.access_token = data['tokens']['access']
            st.session_state.refresh_token = data['tokens']['refresh']
            return True, "Login successful!"
        else:
            error_data = response.json()
            return False, error_data.get('error', 'Login failed')
    except Exception as e:
        return False, f"Connection error: {str(e)}"

def logout_user():
    """Logout user"""
    try:
        headers = get_auth_headers()
        response = requests.post(f"{API_BASE_URL}/auth/logout/", headers=headers)
        
        # Clear session state regardless of response
        st.session_state.authenticated = False
        st.session_state.user = None
        st.session_state.access_token = None
        st.session_state.refresh_token = None
        
        return True, "Logout successful!"
    except Exception as e:
        # Clear session state even if API call fails
        st.session_state.authenticated = False
        st.session_state.user = None
        st.session_state.access_token = None
        st.session_state.refresh_token = None
        return True, "Logout successful!"

def get_user_profile():
    """Get current user profile"""
    try:
        headers = get_auth_headers()
        response = requests.get(f"{API_BASE_URL}/auth/profile/", headers=headers)
        
        if response.status_code == 200:
            return response.json()['user']
        else:
            return None
    except Exception as e:
        return None

def show_login_page():
    """Display login page"""
    st.header("🔐 Login to JV Cinelytics")
    
    with st.form("login_form"):
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        col1, col2 = st.columns(2)
        with col1:
            login_submitted = st.form_submit_button("Login", type="primary")
        with col2:
            st.form_submit_button("Switch to Signup", on_click=lambda: st.session_state.update({'show_signup': True}))
        
        if login_submitted:
            if username and password:
                success, message = login_user(username, password)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
            else:
                st.error("Please enter both username and password")

def show_signup_page():
    """Display signup page"""
    st.header("📝 Create Account")
    
    with st.form("signup_form"):
        username = st.text_input("Username", key="signup_username")
        email = st.text_input("Email", key="signup_email")
        password = st.text_input("Password", type="password", key="signup_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm_password")
        
        col1, col2 = st.columns(2)
        with col1:
            signup_submitted = st.form_submit_button("Sign Up", type="primary")
        with col2:
            st.form_submit_button("Switch to Login", on_click=lambda: st.session_state.update({'show_signup': False}))
        
        if signup_submitted:
            if not all([username, email, password, confirm_password]):
                st.error("Please fill in all fields")
            elif password != confirm_password:
                st.error("Passwords do not match")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters long")
            else:
                success, message = signup_user(username, email, password)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

def show_auth_pages():
    """Show authentication pages (login/signup)"""
    init_session_state()
    
    # Initialize show_signup in session state
    if 'show_signup' not in st.session_state:
        st.session_state.show_signup = False
    
    if st.session_state.show_signup:
        show_signup_page()
    else:
        show_login_page()

def require_auth():
    """Decorator to require authentication for pages"""
    init_session_state()
    
    if not st.session_state.authenticated:
        st.warning("Please login to access this page")
        show_auth_pages()
        st.stop()
    
    return True 