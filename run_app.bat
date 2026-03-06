@echo off
cd /d %~dp0
python train_model.py
python app.py
pause