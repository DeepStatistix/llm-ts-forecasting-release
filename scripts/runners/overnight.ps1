# Wait for the running Qwen job, then smoke-test and run Gemma 4 E4B on all four tasks.
Set-Location E:\llm-ts-forecasting
$py = "E:\envs\tsllm\python.exe"
$log = "results\overnight_log.txt"
function Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Out-File $log -Append -Encoding utf8 }
Log "waiting for Qwen (pid 37960)"
Wait-Process -Id 37960 -ErrorAction SilentlyContinue
Log "Qwen finished; Gemma smoke test"
& $py -m scripts.run_llm gemma4_e4b flu --n 8 > results\gemma_smoke_log.txt 2> results\gemma_smoke_err.txt
if ($LASTEXITCODE -ne 0) { Log "Gemma smoke FAILED (exit $LASTEXITCODE); stopping"; exit 1 }
Log "Gemma smoke ok: $(Select-String -Path results\gemma_smoke_log.txt -Pattern '^== ' | Select-Object -Last 1)"
& $py -m scripts.run_llm gemma4_e4b flu fred wiki eia > results\llm_gemma_log.txt 2> results\llm_gemma_err.txt
Log "Gemma full run exit $LASTEXITCODE"
