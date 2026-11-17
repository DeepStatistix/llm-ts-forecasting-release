# Smoke-test then run Mistral-7B-Instruct-v0.3 on all four tasks, then the same-row comparison.
Set-Location E:\llm-ts-forecasting
$py = "E:\envs\tsllm\python.exe"
$log = "results\overnight_log.txt"
function Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Out-File $log -Append -Encoding utf8 }
Log "Mistral smoke test"
& $py -m scripts.run_llm mistral7b flu --n 8 > results\mistral_smoke_log.txt 2> results\mistral_smoke_err.txt
if ($LASTEXITCODE -ne 0) { Log "Mistral smoke FAILED (exit $LASTEXITCODE)"; exit 1 }
Log "Mistral smoke ok: $(Select-String -Path results\mistral_smoke_log.txt -Pattern '^== ' | Select-Object -Last 1)"
& $py -m scripts.run_llm mistral7b flu fred wiki eia > results\llm_mistral_log.txt 2> results\llm_mistral_err.txt
Log "Mistral full run exit $LASTEXITCODE"
& $py -m scripts.compare_llm > results\compare_llm_log.txt 2> results\compare_llm_err.txt
Log "compare_llm exit $LASTEXITCODE"
