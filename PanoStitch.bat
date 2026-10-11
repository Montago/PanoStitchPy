@echo off
rem Starter PanoStitch uden et konsolvindue. Dobbeltklik på filen, eller lav en genvej til den på skrivebordet.
python -c "import PySide6, cv2, numpy, scipy, PIL" >nul 2>nul
if errorlevel 1 (
    echo.
    echo  PanoStitch mangler Python eller nogle pakker.
    echo  Koer Installer.bat foerst.
    echo.
    pause
    exit /b 1
)
start "" pythonw "%~dp0app\main.py" %*
