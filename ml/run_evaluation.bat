@echo off
REM ============================================================================
REM JV Cinelytics - Model Evaluation Script
REM Evaluates trained model and generates confusion matrices
REM ============================================================================

echo ========================================
echo JV Cinelytics - Model Evaluation
echo ========================================
echo.

REM Activate virtual environment
if exist ..\venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call ..\venv\Scripts\activate.bat
) else (
    echo Warning: Virtual environment not found. Using global Python.
)

REM Check if required packages are installed
echo Checking dependencies...
python -c "import matplotlib, seaborn, sklearn" 2>nul
if errorlevel 1 (
    echo Installing required packages...
    pip install matplotlib seaborn scikit-learn -q
)

echo.
echo ========================================
echo Running Evaluation...
echo ========================================
echo.

REM Run evaluation
REM Modify these paths as needed:
set CHECKPOINT=runs\exp1\model.pt
set VAL_DATA=..\data\val.jsonl
set OUTPUT_DIR=evaluation_results

python evaluate.py --checkpoint %CHECKPOINT% --val %VAL_DATA% --output_dir %OUTPUT_DIR%

if errorlevel 1 (
    echo.
    echo ========================================
    echo ERROR: Evaluation failed!
    echo ========================================
    echo.
    echo Possible reasons:
    echo 1. Model checkpoint not found at: %CHECKPOINT%
    echo 2. Validation data not found at: %VAL_DATA%
    echo 3. Missing dependencies
    echo.
    echo Please check the paths and try again.
    echo See EVALUATION_GUIDE.md for help.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Evaluation Complete!
echo ========================================
echo.
echo Results saved to: %OUTPUT_DIR%
echo.
echo Generated files:
echo   - Confusion matrices (PNG images)
echo   - Metrics summary (JSON)
echo   - Comparison plot (PNG)
echo.
echo Opening results folder...
explorer %OUTPUT_DIR%

pause
