@echo off
cd /d E:\llm-ts-forecasting
set SKIP=arima
E:\envs\tsllm\python.exe -m scripts.export_task fred
E:\envs\tsllm_moirai\Scripts\python.exe -I scripts\run_moirai.py fred
E:\envs\tsllm\python.exe -m scripts.run_baselines fred
E:\envs\tsllm\python.exe -m scripts.run_foundation chronos_bolt_small chronos_bolt_base chronos2 timesfm25 moirai2 nhits patchtst
E:\envs\tsllm\python.exe -m scripts.run_tuned fred
echo ALLDONE
