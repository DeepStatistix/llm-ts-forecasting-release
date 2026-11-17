@echo off
cd /d E:\llm-ts-forecasting
E:\envs\tsllm\python.exe -m scripts.export_task m4_monthly m4_weekly m4_daily
E:\envs\tsllm\python.exe -m scripts.run_anchor m4_monthly m4_weekly m4_daily
E:\envs\tsllm_moirai\Scripts\python.exe -I scripts\run_moirai.py m4_monthly m4_weekly m4_daily
E:\envs\tsllm\python.exe -m scripts.run_anchor m4_monthly m4_weekly m4_daily
echo ANCHORDONE
