@echo off
chcp 65001 >nul 2>&1
title Install Dependencies
cd /d "%~dp0"

if not exist "logs" mkdir logs
set LOG=logs\install_%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%.txt
set LOG=%LOG: =0%

echo.
echo ========================================
echo   Installing Dependencies
echo ========================================
echo.

set PYTHON=
for /f "delims=" %%i in ('where python 2^>nul') do (
    if not defined PYTHON set PYTHON=%%i
)
if not defined PYTHON (
    echo [ERROR] Python not found!
    goto :end
)
echo Python: %PYTHON%

:: Detect GPU type
echo.
echo [GPU] Detecting graphics card...
set GPU_TYPE=CPU

:: Check NVIDIA
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    set GPU_TYPE=NVIDIA
    echo       Found NVIDIA GPU - will use CUDA
    goto :gpu_done
)

:: Check AMD via WMIC
for /f "tokens=2 delims==" %%a in ('wmic path win32_videocontroller get name /value 2^>nul ^| find "="') do (
    echo %%a | findstr /i "AMD Radeon" >nul
    if not errorlevel 1 (
        set GPU_TYPE=AMD
        set GPU_NAME=%%a
    )
    echo %%a | findstr /i "NVIDIA" >nul
    if not errorlevel 1 (
        set GPU_TYPE=NVIDIA
    )
)

if "%GPU_TYPE%"=="AMD" (
    echo       Found AMD GPU: %GPU_NAME%
    echo       Will install torch-directml for GPU acceleration
) else if "%GPU_TYPE%"=="NVIDIA" (
    echo       Found NVIDIA GPU - will use CUDA
) else (
    echo       No discrete GPU found - will use CPU
)

:gpu_done
echo [%date% %time%] GPU: %GPU_TYPE% >> %LOG%
echo.

:: Install step by step
echo [1/6] Flask...
%PYTHON% -c "import flask" 2>nul && echo       OK || (%PYTHON% -m pip install flask >> %LOG% 2>&1 && echo       OK || echo       FAILED)

echo [2/6] Numpy...
%PYTHON% -c "import numpy" 2>nul && echo       OK || (%PYTHON% -m pip install numpy >> %LOG% 2>&1 && echo       OK || echo       FAILED)

echo [3/6] Torch...
%PYTHON% -c "import torch" 2>nul && echo       OK || (
    if "%GPU_TYPE%"=="NVIDIA" (
        echo       Installing torch with CUDA...
        %PYTHON% -m pip install torch torchaudio >> %LOG% 2>&1
    ) else (
        echo       Installing torch CPU version...
        %PYTHON% -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu >> %LOG% 2>&1
    )
    %PYTHON% -c "import torch" 2>nul && echo       OK || echo       FAILED - check %LOG%
)

:: Install DirectML for AMD
if "%GPU_TYPE%"=="AMD" (
    echo [3b] torch-directml for AMD GPU...
    %PYTHON% -c "import torch_directml" 2>nul && echo       OK || (
        %PYTHON% -m pip install torch-directml >> %LOG% 2>&1
        %PYTHON% -c "import torch_directml" 2>nul && echo       OK || echo       FAILED - CPU fallback
    )
)

echo [4/6] Audio libs...
%PYTHON% -c "import librosa" 2>nul && echo       OK || (
    %PYTHON% -m pip install librosa soundfile scipy >> %LOG% 2>&1
    echo       OK
)

echo [5/6] ONNX...
%PYTHON% -c "import onnxruntime" 2>nul && echo       OK || (
    if "%GPU_TYPE%"=="AMD" (
        echo       Installing onnxruntime-directml...
        %PYTHON% -m pip install onnxruntime-directml >> %LOG% 2>&1
    )
    %PYTHON% -m pip install onnxruntime >> %LOG% 2>&1
    echo       OK
)

echo [6/6] Others...
%PYTHON% -c "import pypinyin" 2>nul || %PYTHON% -m pip install pypinyin >> %LOG% 2>&1
%PYTHON% -c "import onnxscript" 2>nul || %PYTHON% -m pip install onnxscript >> %LOG% 2>&1
echo       Done

echo.
echo ========================================
echo   All done! GPU mode: %GPU_TYPE%
echo   Run run.bat to start.
echo ========================================
echo.

:end
pause
