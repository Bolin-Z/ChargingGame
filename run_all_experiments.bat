@echo off
chcp 65001 >nul
echo ========================================
echo 启动9个并行实验 (3算法 x 3场景)
echo ========================================
echo.

set SEED=42

:: MADDPG 实验
start "MADDPG_SiouxFalls" cmd /k "conda activate drl && python run_single_experiment.py -a maddpg -s siouxfalls --seed %SEED%"
timeout /t 2 /nobreak >nul
start "MADDPG_Berlin" cmd /k "conda activate drl && python run_single_experiment.py -a maddpg -s berlin --seed %SEED%"
timeout /t 2 /nobreak >nul
start "MADDPG_Anaheim" cmd /k "conda activate drl && python run_single_experiment.py -a maddpg -s anaheim --seed %SEED%"
timeout /t 2 /nobreak >nul

:: IDDPG 实验
start "IDDPG_SiouxFalls" cmd /k "conda activate drl && python run_single_experiment.py -a iddpg -s siouxfalls --seed %SEED%"
timeout /t 2 /nobreak >nul
start "IDDPG_Berlin" cmd /k "conda activate drl && python run_single_experiment.py -a iddpg -s berlin --seed %SEED%"
timeout /t 2 /nobreak >nul
start "IDDPG_Anaheim" cmd /k "conda activate drl && python run_single_experiment.py -a iddpg -s anaheim --seed %SEED%"
timeout /t 2 /nobreak >nul

:: MFDDPG 实验
start "MFDDPG_SiouxFalls" cmd /k "conda activate drl && python run_single_experiment.py -a mfddpg -s siouxfalls --seed %SEED%"
timeout /t 2 /nobreak >nul
start "MFDDPG_Berlin" cmd /k "conda activate drl && python run_single_experiment.py -a mfddpg -s berlin --seed %SEED%"
timeout /t 2 /nobreak >nul
start "MFDDPG_Anaheim" cmd /k "conda activate drl && python run_single_experiment.py -a mfddpg -s anaheim --seed %SEED%"

echo 已启动9个实验窗口
echo.
echo 提示: 使用 Win+Z 可快速将窗口排列为3x3网格
echo.
pause
