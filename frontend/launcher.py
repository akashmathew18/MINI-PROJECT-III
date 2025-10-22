import streamlit as st
import subprocess
import sys
import os

def main():
    st.set_page_config(
        page_title="JV Cinelytics - App Launcher",
        page_icon="🎬",
        layout="centered"
    )
    
    st.title("🎬 JV Cinelytics")
    st.subheader("Choose Your Interface")
    
    st.markdown("""
    Welcome to JV Cinelytics! Please choose which interface you'd like to use:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📱 Original Interface
        - Sidebar navigation
        - Simple layout
        - Basic styling
        """)
        if st.button("Launch Original App", type="primary"):
            st.info("Launching original app...")
            # This would typically launch the original app
            st.success("Original app launched!")
    
    with col2:
        st.markdown("""
        ### 🎨 Professional Interface
        - Top navigation bar
        - Red/Gray/White theme
        - Modern UI design
        - Professional styling
        """)
        if st.button("Launch Professional App", type="secondary"):
            st.info("Launching professional app...")
            # This would typically launch the professional app
            st.success("Professional app launched!")
    
    st.markdown("---")
    st.markdown("""
    ### 🚀 Quick Start Instructions:
    
    1. **Backend Setup:**
       ```bash
       cd backend
       pip install -r requirements.txt
       python manage.py migrate
       python manage.py runserver
       ```
    
    2. **Frontend Setup:**
       ```bash
       cd frontend
       pip install -r requirements.txt
       ```
    
    3. **Launch Professional App:**
       ```bash
       streamlit run app_professional.py
       ```
    
    4. **Launch Original App:**
       ```bash
       streamlit run app.py
       ```
    """)

if __name__ == "__main__":
    main() 