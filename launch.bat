@echo off
echo Starting JV Cinelytics - Complete ML & Script Analysis Platform...

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt

REM Setup admin user (if not exists)
echo Setting up admin user...
python setup_admin.py

REM Launch the unified app
echo Launching JV Cinelytics...
streamlit run app.py

pause
