@echo off
echo ====================================================
echo  ADTC 2026: Setting up Offline Smart Farm Assistant
echo ====================================================

echo [1/5] Creating Python Virtual Environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo Error creating virtual environment. Is Python installed?
    pause
    exit /b
)

echo [2/5] Activating Virtual Environment...
call venv\Scripts\activate

echo [3/5] Upgrading pip...
python -m pip install --upgrade pip

echo [4/5] Installing CPU-Optimized Llama engine (Saves 4GB RAM)...
pip install llama-cpp-python==0.2.56 --extra-index-url https://github.io

echo [5/5] Installing remaining project requirements...
pip install -r requirements.txt

echo ====================================================
echo  Setup Complete! Launching Streamlit Application...
echo ====================================================
streamlit run app.py
pause
