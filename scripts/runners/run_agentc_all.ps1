# Waits for the prompted text-context chain, then reruns the agent with CONSTRAINED decoding
# (plain + text context, both LLMs), then the prompt-sensitivity test with constrained decoding.
Set-Location E:\llm-ts-forecasting
$py = "E:\envs\tsllm\python.exe"
$log = "results\overnight_log.txt"
function Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Out-File $log -Append -Encoding utf8 }
while (-not (Select-String -Path $log -Pattern "ctx ALL DONE" -Quiet)) { Start-Sleep 30 }
foreach ($m in @("qwen3_8b","mistral7b")) {
  Log "agentc $m plain start"
  & $py -m scripts.run_agent $m > "results\agentc_${m}_log.txt" 2> "results\agentc_${m}_err.txt"
  Log "agentc $m plain exit $LASTEXITCODE"
  & $py -m scripts.run_agent $m --ctx > "results\agentcctx_${m}_log.txt" 2> "results\agentcctx_${m}_err.txt"
  Log "agentc $m ctx exit $LASTEXITCODE"
}
foreach ($m in @("qwen3_8b","mistral7b")) {
  & $py -m scripts.agent_sensitivity $m > "results\sensc_${m}_log.txt" 2> "results\sensc_${m}_err.txt"
  Log "agentc sensitivity $m exit $LASTEXITCODE"
}
Log "agentc ALL DONE"
