Set-Location E:\llm-ts-forecasting
$py = "E:\envs\tsllm\python.exe"
$log = "results\overnight_log.txt"
function Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Out-File $log -Append -Encoding utf8 }
Log "agent qwen3_8b start"
& $py -m scripts.run_agent qwen3_8b > results\agent_qwen_log.txt 2> results\agent_qwen_err.txt
Log "agent qwen3_8b exit $LASTEXITCODE"
& $py -m scripts.run_agent mistral7b > results\agent_mistral_log.txt 2> results\agent_mistral_err.txt
Log "agent mistral7b exit $LASTEXITCODE"
