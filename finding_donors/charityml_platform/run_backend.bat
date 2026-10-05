@echo off
title CharityML Backend API (Flask)
echo ========================================================
echo   CharityML Machine Learning Backend Engine
echo   Starting Flask REST API on http://127.0.0.1:5000
echo ========================================================
cd backend
python -m pip install -r requirements.txt
python app.py
pause
