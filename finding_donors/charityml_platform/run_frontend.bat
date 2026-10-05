@echo off
title CharityML Modern Reflex Dashboard
echo ========================================================
echo   CharityML Intelligence Platform (Reflex UI)
echo   Starting Dashboard on http://localhost:3000
echo ========================================================
set "PATH=%PATH%;C:\Program Files\nodejs"
cd frontend
reflex run
pause
