@echo off
chcp 65001 >nul
echo.
echo  PanoStitch - installation af de pakker programmet skal bruge
echo  ============================================================
echo.
python -c "import sys" >nul 2>nul
if errorlevel 1 (
    echo  Python er ikke installeret.
    echo.
    echo  Installer Python 3.13 fra https://www.python.org/downloads/
    echo  ^(husk at saette kryds ved "Add python.exe to PATH"^)
    echo  eller koer denne kommando i en terminal:
    echo.
    echo      winget install Python.Python.3.13
    echo.
    echo  Koer derefter Installer.bat igen.
    echo.
    pause
    exit /b 1
)
python -m pip install --upgrade -r "%~dp0requirements.txt"
if errorlevel 1 (
    echo.
    echo  Installationen fejlede - se beskeden ovenfor.
    pause
    exit /b 1
)
echo.
echo  Faerdig! Start programmet med PanoStitch.bat
echo.
pause
