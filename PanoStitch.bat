@echo off
rem Starts PanoStitch without a console window. Double-click this file, or make a desktop shortcut to it.
python -c "import PySide6, cv2, numpy, scipy, PIL" >nul 2>nul
if errorlevel 1 (
    echo.
    echo  PanoStitch is missing Python or some packages.
    echo  Run Installer.bat first.
    echo.
    pause
    exit /b 1
)
start "" pythonw "%~dp0app\main.py" %*
