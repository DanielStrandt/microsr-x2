@echo off
setlocal
cd /d "%~dp0"
python microsr_gui.py
if errorlevel 1 (
  echo.
  echo MicroSR GUI exited with an error. Check that Python, PyTorch, Pillow and NumPy are installed.
  pause
)

