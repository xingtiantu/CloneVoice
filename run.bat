@echo off
title Voice Clone Trainer
cd /d "%~dp0"

if not exist "logs" mkdir logs
set TS=%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%
set TS=%TS: =0%
set LOG=logs\run_%TS%.txt

set PYTHONIOENCODING=utf-8
set PYTHON=
for /f "delims=" %%i in ('where python 2^>nul') do (
    if not defined PYTHON set PYTHON=%%i
)
if not defined PYTHON (
    echo [ERROR] Python not found!
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Voice Clone Trainer v1.0
echo ========================================
echo.

:: --- Auto-check and install missing deps ---
echo Checking dependencies...
set NEED_INSTALL=0

"%PYTHON%" -c "import flask" 2>nul
if %errorlevel% neq 0 set NEED_INSTALL=1
"%PYTHON%" -c "import torch" 2>nul
if %errorlevel% neq 0 set NEED_INSTALL=1
"%PYTHON%" -c "import librosa" 2>nul
if %errorlevel% neq 0 set NEED_INSTALL=1
"%PYTHON%" -c "import onnxruntime" 2>nul
if %errorlevel% neq 0 set NEED_INSTALL=1
"%PYTHON%" -c "import pypinyin" 2>nul
if %errorlevel% neq 0 set NEED_INSTALL=1
"%PYTHON%" -c "import onnxscript" 2>nul
if %errorlevel% neq 0 set NEED_INSTALL=1
"%PYTHON%" -c "import whisper" 2>nul
if %errorlevel% neq 0 set NEED_INSTALL=1

if %NEED_INSTALL%==1 (
    echo       Some packages missing, installing...
    echo [%date% %time%] Auto-installing missing deps >> %LOG%
    call install.bat
) else (
    echo       All OK
)

:: --- Show device info ---
echo.
echo [Device] PyTorch info:
"%PYTHON%" -c "import torch; print('  Torch:', torch.__version__)"
"%PYTHON%" -c "import torch; print('  CUDA:', torch.cuda.is_available())"
"%PYTHON%" -c "import torch_directml; print('  DirectML:', torch_directml.is_available()); print('  GPU:', torch_directml.device_name(0) if torch_directml.is_available() else 'CPU')"

if not exist "uploads" mkdir uploads
if not exist "models" mkdir models

echo.
echo ========================================
echo   http://127.0.0.1:5000
echo   Logs: logs\
echo   Ctrl+C to stop
echo ========================================
echo.

start "" cmd /c "timeout /t 3 >nul && start http://127.0.0.1:5000"

"%PYTHON%" server.py >> %LOG% 2>&1
echo.
echo [STOP] Exit code: %errorlevel%
echo --- Last 15 lines ---
powershell -Command "Get-Content '%LOG%' -Tail 15"

echo.
pause
