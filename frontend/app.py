import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# API Configuration
API_BASE_URL = "http://localhost:8000/api"

def main():
    st.set_page_config(
        page_title="JV Cinelytics - Script Analysis",
        page_icon="🎬",
        layout="wide"
    )

    # Main app interface
    st.title("🎬 JV Cinelytics")
    st.subheader("Intelligent Movie Script Analysis")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Dashboard", "Script Analysis", "My Scripts", "TTS Narration", "Profile", "About"]
    )

    if page == "Dashboard":
        show_dashboard()
    elif page == "Script Analysis":
        show_script_analysis()
    elif page == "My Scripts":
        show_my_scripts()
    elif page == "TTS Narration":
        show_tts_narration()
    elif page == "Profile":
        show_profile()
    elif page == "About":
        show_about_page()

def show_dashboard():
    """Show user dashboard with statistics"""
    st.header("📊 Dashboard")
    
    try:
        response = requests.get(f"{API_BASE_URL}/scripts/")
        
        if response.status_code == 200:
            scripts = response.json().get('scripts', [])
            
            if scripts:
                total_scripts = len(scripts)
                total_scenes = sum(script['scene_count'] for script in scripts)
                total_characters = sum(script['character_count'] for script in scripts)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Scripts", total_scripts)
                with col2:
                    st.metric("Total Scenes", total_scenes)
                with col3:
                    st.metric("Total Characters", total_characters)
                with col4:
                    avg_scenes = total_scenes / total_scripts if total_scripts > 0 else 0
                    st.metric("Avg Scenes/Script", f"{avg_scenes:.1f}")
                
                st.subheader("Recent Scripts")
                recent_scripts = scripts[:5]
                
                for script in recent_scripts:
                    with st.expander(f"{script['title']} ({script['uploaded_at'][:10]})"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"**Genre:** {script['genre']}")
                        with col2:
                            st.write(f"**Sentiment:** {script['sentiment']}")
                        with col3:
                            st.write(f"**Characters:** {script['character_count']}")
            else:
                st.info("No scripts uploaded yet.")
        else:
            st.error("Failed to load scripts")
    except Exception as e:
        st.error(f"Error loading dashboard: {str(e)}")

def show_script_analysis():
    """Show script analysis page"""
    st.header("Script Analysis")
    
    uploaded_file = st.file_uploader(
        "Upload your script file (.txt)",
        type=['txt'],
        help="Upload a movie script in .txt format"
    )
    
    if uploaded_file is not None:
        script_content = uploaded_file.read().decode('utf-8')
        st.text_area("Script Preview", script_content, height=200)
        
        script_title = st.text_input("Script Title", value=uploaded_file.name.replace('.txt', ''))
        
        col1, col2 = st.columns(2)
        with col1:
            analyze_scenes = st.checkbox("Scene Breakdown", value=True)
            analyze_characters = st.checkbox("Character Extraction", value=True)
        with col2:
            analyze_sentiment = st.checkbox("Sentiment Analysis")
            analyze_genre = st.checkbox("Genre Classification")
        
        if st.button("Upload & Analyze Script", type="primary"):
            with st.spinner("Uploading and analyzing script..."):
                try:
                    response = requests.post(f"{API_BASE_URL}/scripts/upload/", 
                        json={
                            'script': script_content,
                            'title': script_title
                        })
                    
                    if response.status_code == 201:
                        data = response.json()
                        st.success("Script uploaded and analyzed successfully!")
                        
                        analysis = data['analysis']
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Sentiment", analysis['sentiment'])
                        with col2:
                            st.metric("Genre", analysis['genre'])
                        with col3:
                            st.metric("Characters", analysis['character_count'])
                        with col4:
                            st.metric("Scenes", analysis['scene_count'])
                        
                        st.info(f"Script ID: {data['script_id']}")
                    else:
                        st.error(f"Upload failed: {response.text}")
                except Exception as e:
                    st.error(f"Error uploading script: {str(e)}")

def show_my_scripts():
    """Show user's uploaded scripts"""
    st.header("My Scripts")
    
    try:
        response = requests.get(f"{API_BASE_URL}/scripts/")
        
        if response.status_code == 200:
            scripts = response.json().get('scripts', [])
            
            if scripts:
                df = pd.DataFrame(scripts)
                df['uploaded_at'] = pd.to_datetime(df['uploaded_at']).dt.strftime('%Y-%m-%d')
                
                st.dataframe(
                    df[['title', 'genre', 'sentiment', 'character_count', 'scene_count', 'uploaded_at']],
                    use_container_width=True
                )
                
                st.subheader("Script Details")
                selected_script = st.selectbox(
                    "Select a script to view details",
                    options=scripts,
                    format_func=lambda x: f"{x['title']} ({x['uploaded_at'][:10]})"
                )
                
                if selected_script:
                    with st.expander(f"Details for {selected_script['title']}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Title:** {selected_script['title']}")
                            st.write(f"**Genre:** {selected_script['genre']}")
                            st.write(f"**Sentiment:** {selected_script['sentiment']}")
                        with col2:
                            st.write(f"**Characters:** {selected_script['character_count']}")
                            st.write(f"**Scenes:** {selected_script['scene_count']}")
                            st.write(f"**Uploaded:** {selected_script['uploaded_at']}")
            else:
                st.info("No scripts uploaded yet.")
        else:
            st.error("Failed to load scripts")
    except Exception as e:
        st.error(f"Error loading scripts: {str(e)}")

def show_tts_narration():
    """Show TTS narration page"""
    st.header("Text-to-Speech Narration")
    
    text = st.text_area("Enter text to narrate:", height=100)
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Voice Gender", ["male", "female"])
    with col2:
        emotion = st.selectbox("Voice Emotion", ["neutral", "angry", "cheerful", "sad"])
    
    if st.button("Generate Narration", type="primary"):
        if text.strip():
            with st.spinner("Generating audio..."):
                try:
                    response = requests.post(f"{API_BASE_URL}/tts-narration/", 
                        json={"text": text, "gender": gender, "emotion": emotion})
                    if response.status_code == 200:
                        audio_file = response.json()['audio_file']
                        st.success(f"Audio generated: {audio_file}")
                        st.audio(audio_file, format='audio/mp3')
                    else:
                        st.error(f"TTS failed: {response.text}")
                except Exception as e:
                    st.error(f"Error in TTS: {str(e)}")
        else:
            st.warning("Please enter some text to narrate.")

def show_profile():
    """Show user profile page (no auth, just placeholder)"""
    st.header("👤 User Profile")
    st.write("Profile details would appear here without authentication.")

def show_about_page():
    """Show about page"""
    st.header("About JV Cinelytics")
    st.write("""
    **JV Cinelytics** is a web-based application that leverages Natural Language Processing (NLP) 
    to assist filmmakers, scriptwriters, and producers in analyzing movie scripts.
    
    ### Technology Stack:
    - **Frontend**: Streamlit (Python)
    - **Backend**: Django + Django REST Framework
    - **NLP**: PyTorch, Custom Models
    - **TTS**: Edge-TTS (Microsoft)
    - **Architecture**: Modular, API-first
    
    ### Features:
    - Scene breakdown using regex patterns
    - Character extraction from ALL CAPS names
    - Sentiment analysis using PyTorch models
    - Genre classification for 7 major genres
    - Text-to-speech with gender and emotion support
    - Script management and analytics
    """)

if __name__ == "__main__":
    main()
