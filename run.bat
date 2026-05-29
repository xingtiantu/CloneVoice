@echo off
chcp 65001 >nul 2>&1
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
    goto :end
)

echo.
echo ========================================
echo   Voice Clone Trainer v1.0
echo ========================================
echo.

:: Auto-check and install missing deps
echo Checking dependencies...
set NEED_INSTALL=0

%PYTHON% -c "import flask" 2>nul || (set NEED_INSTALL=1)
%PYTHON% -c "import torch" 2>nul || (set NEED_INSTALL=1)
%PYTHON% -c "import librosa" 2>nul || (set NEED_INSTALL=1)
%PYTHON% -c "import onnxruntime" 2>nul || (set NEED_INSTALL=1)
%PYTHON% -c "import pypinyin" 2>nul || (set NEED_INSTALL=1)
%PYTHON% -c "import onnxscript" 2>nul || (set NEED_INSTALL=1)

if %NEED_INSTALL%==1 (
    echo       Some packages missing, installing...
    echo [%date% %time%] Auto-installing missing deps >> %LOG%
    call install.bat
) else (
    echo       All OK
)

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

%PYTHON% server.py >> %LOG% 2>&1
echo.
echo [STOP] Exit code: %errorlevel%
echo --- Last 15 lines ---
powershell -Command "Get-Content '%LOG%' -Tail 15"

:end
echo.
pause
