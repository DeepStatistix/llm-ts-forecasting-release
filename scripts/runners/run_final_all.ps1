# Final experiment queue (runs after the constrained sensitivity test): seed variance, ARIMA on wiki.
Set-Location E:\llm-ts-forecasting
$py = "E:\envs\tsllm\python.exe"
$log = "results\overnight_log.txt"
function Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Out-File $log -Append -Encoding utf8 }
while (-not (Select-String -Path $log -Pattern "agentc ALL DONE" -Quiet)) { Start-Sleep 30 }
Log "final: deep seeds start"
& $py -m scripts.seeds_deep > results\seeds_deep_log.txt 2> results\seeds_deep_err.txt
Log "final: deep seeds exit $LASTEXITCODE"
foreach ($s in @(1,2)) {
  foreach ($t in @(@("flu","100"), @("fred","100"), @("wiki","40"))) {
    & $py -m scripts.run_llm qwen3_8b $t[0] --seed $s --rows $t[1] >> results\seeds_qwen_log.txt 2>> results\seeds_qwen_err.txt
    Log "final: qwen seed $s $($t[0]) exit $LASTEXITCODE"
  }
}
foreach ($s in @(1,2)) {
  foreach ($t in @(@("flu","100"), @("fred","100"))) {
    & $py -m scripts.run_llm mistral7b $t[0] --seed $s --rows $t[1] >> results\seeds_mistral_log.txt 2>> results\seeds_mistral_err.txt
    Log "final: mistral seed $s $($t[0]) exit $LASTEXITCODE"
  }
}
Log "final: arima wiki start"
& $py -m scripts.run_arima_wiki > results\arima_wiki_log.txt 2> results\arima_wiki_err.txt
Log "final: arima wiki exit $LASTEXITCODE"
Log "final ALL DONE"
