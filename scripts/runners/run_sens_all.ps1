Set-Location E:\llm-ts-forecasting
$py = "E:\envs\tsllm\python.exe"
$log = "results\overnight_log.txt"
function Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Out-File $log -Append -Encoding utf8 }
Log "agent sensitivity qwen3_8b start"
& $py -m scripts.agent_sensitivity qwen3_8b > results\sens_qwen_log.txt 2> results\sens_qwen_err.txt
Log "agent sensitivity qwen3_8b exit $LASTEXITCODE"
& $py -m scripts.agent_sensitivity mistral7b > results\sens_mistral_log.txt 2> results\sens_mistral_err.txt
Log "agent sensitivity mistral7b exit $LASTEXITCODE"
