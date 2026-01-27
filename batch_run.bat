@echo off
REM ============================================================
REM 批量实验运行脚本
REM
REM 用法：
REM     batch_run.bat                     运行所有预设实验
REM     batch_run.bat --parallel          并行运行（每个实验一个窗口）
REM     batch_run.bat --algo MADDPG       只运行 MADDPG 算法
REM     batch_run.bat --network siouxfalls 只运行 siouxfalls 网络
REM ============================================================

setlocal enabledelayedexpansion

REM 激活 conda 环境
call conda activate drl
if errorlevel 1 (
    echo [错误] 无法激活 conda 环境 'drl'
    echo 请确保已创建 drl 环境：conda create -n drl python=3.10
    exit /b 1
)

REM 切换到项目目录
cd /d %~dp0

REM 解析参数
set PARALLEL=0
set FILTER_ALGO=
set FILTER_NETWORK=

:parse_args
if "%~1"=="" goto end_parse
if "%~1"=="--parallel" (
    set PARALLEL=1
    shift
    goto parse_args
)
if "%~1"=="--algo" (
    set FILTER_ALGO=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="--network" (
    set FILTER_NETWORK=%~2
    shift
    shift
    goto parse_args
)
shift
goto parse_args
:end_parse

echo ============================================================
echo 批量实验运行
echo ============================================================
echo 并行模式: %PARALLEL%
if not "%FILTER_ALGO%"=="" echo 算法过滤: %FILTER_ALGO%
if not "%FILTER_NETWORK%"=="" echo 网络过滤: %FILTER_NETWORK%
echo ============================================================

REM 定义实验配置
set NETWORKS=siouxfalls
set ALGOS=MADDPG IDDPG MFDDPG
set SEEDS=42

REM 计数器
set COUNT=0

REM 遍历所有组合
for %%N in (%NETWORKS%) do (
    REM 检查网络过滤
    if not "%FILTER_NETWORK%"=="" (
        if not "%%N"=="%FILTER_NETWORK%" goto skip_network
    )

    for %%A in (%ALGOS%) do (
        REM 检查算法过滤
        if not "%FILTER_ALGO%"=="" (
            if not "%%A"=="%FILTER_ALGO%" goto skip_algo
        )

        for %%S in (%SEEDS%) do (
            set /a COUNT+=1
            echo.
            echo [!COUNT!] 运行: %%N / %%A / seed=%%S

            if %PARALLEL%==1 (
                REM 并行模式：在新窗口中运行
                start "%%N_%%A_%%S" cmd /c "conda activate drl && python run_experiment.py --network %%N --algo %%A --seed %%S --workers 8 --no-monitor"
            ) else (
                REM 串行模式：在当前窗口运行
                python run_experiment.py --network %%N --algo %%A --seed %%S --workers 8 --no-monitor
                if errorlevel 1 (
                    echo [警告] 实验 %%N/%%A/%%S 未收敛
                )
            )
        )
        :skip_algo
    )
    :skip_network
)

echo.
echo ============================================================
echo 批量实验完成，共运行 %COUNT% 个实验
echo ============================================================

endlocal
