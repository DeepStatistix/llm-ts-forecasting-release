Set-Location E:\llm-ts-forecasting
$py = "E:\envs\tsllm\python.exe"
$log = "results\overnight_log.txt"
function Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Out-File $log -Append -Encoding utf8 }
Log "ctx agent qwen3_8b start"
& $py -m scripts.run_agent qwen3_8b --ctx > results\ctx_agent_qwen_log.txt 2> results\ctx_agent_qwen_err.txt
Log "ctx agent qwen3_8b exit $LASTEXITCODE"
& $py -m scripts.run_agent mistral7b --ctx > results\ctx_agent_mistral_log.txt 2> results\ctx_agent_mistral_err.txt
Log "ctx agent mistral7b exit $LASTEXITCODE"
foreach ($t in @(@("flu","150"), @("fred","100"), @("wiki","60"))) {
  & $py -m scripts.run_llm qwen3_8b $t[0] --ctx $t[1] >> results\ctx_prompt_qwen_log.txt 2>> results\ctx_prompt_qwen_err.txt
  Log "ctx prompted qwen3_8b $($t[0]) exit $LASTEXITCODE"
}
Log "ctx ALL DONE"
