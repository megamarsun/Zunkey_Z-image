@echo off
REM Nuitka build script for Windows (GUI / onefile)
REM Run in the same venv where requirements are installed.

python -m pip install --upgrade pip
python -m pip install nuitka

python -m nuitka ^
  --standalone ^
  --onefile ^
  --enable-plugin=tk-inter ^
  --windows-disable-console ^
  Zunkey_Z_image_v1_0_0.py

echo.
echo Build finished. Check the dist/ folder.
pause
