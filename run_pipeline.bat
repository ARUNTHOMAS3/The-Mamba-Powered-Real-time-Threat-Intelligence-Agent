@echo off
echo [PIPELINE] Starting CICIDS2017 Tabular Comparison Pipeline...

echo [1/5] Training LSTM (Tabular)...
python train/train_lstm_tabular.py
if %errorlevel% neq 0 (
    echo [ERROR] LSTM Training Failed!
    exit /b %errorlevel%
)

echo [2/5] Training Mamba (Tabular)...
python train/train_mamba_tabular.py
if %errorlevel% neq 0 (
    echo [ERROR] Mamba Training Failed!
    exit /b %errorlevel%
)

echo [3/5] Running Offline Evaluation...
python train/eval_offline.py
if %errorlevel% neq 0 (
    echo [ERROR] Offline Eval Failed!
    exit /b %errorlevel%
)

echo [4/5] Running Streaming Evaluation...
python train/eval_streaming.py
if %errorlevel% neq 0 (
    echo [ERROR] Streaming Eval Failed!
    exit /b %errorlevel%
)

echo [5/5] Generating Tables...
python train/auto_generate_tables.py
if %errorlevel% neq 0 (
    echo [ERROR] Table Generation Failed!
    exit /b %errorlevel%
)

echo [PIPELINE] COMPLETED SUCCESSFULLY!
pause
