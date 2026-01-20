@echo off
chcp 65001 >nul
echo ========================================
echo 启动3个并行实验 (3算法 x SF场景)
echo ========================================
echo.

set SEED=42

:: 三算法并行
start "MADDPG_SiouxFalls" cmd /k "conda activate drl && python run_single_experiment.py -a maddpg -s siouxfalls --seed %SEED%"
timeout /t 2 /nobreak >nul
start "IDDPG_SiouxFalls" cmd /k "conda activate drl && python run_single_experiment.py -a iddpg -s siouxfalls --seed %SEED%"
timeout /t 2 /nobreak >nul
start "MFDDPG_SiouxFalls" cmd /k "conda activate drl && python run_single_experiment.py -a mfddpg -s siouxfalls --seed %SEED%"

echo 已启动3个实验窗口
echo.
echo 提示: 使用 Win+Z 可快速将窗口排列
echo.
pause
