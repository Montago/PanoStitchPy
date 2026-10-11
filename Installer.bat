@echo off
echo.
echo  PanoStitch - installing the packages the program needs
echo  ======================================================
echo.
python -c "import sys" >nul 2>nul
if errorlevel 1 (
    echo  Python is not installed.
    echo.
    echo  Install Python 3.13 from https://www.python.org/downloads/
    echo  ^(tick "Add python.exe to PATH" during the installation^)
    echo  or run this command in a terminal:
    echo.
    echo      winget install Python.Python.3.13
    echo.
    echo  Then run Installer.bat again.
    echo.
    pause
    exit /b 1
)
python -m pip install --upgrade -r "%~dp0requirements.txt"
if errorlevel 1 (
    echo.
    echo  The installation failed - see the message above.
    pause
    exit /b 1
)
echo.
echo  Done! Start the program with PanoStitch.bat
echo.
pause
