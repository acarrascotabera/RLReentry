# Follow-up to campaign.ps1 (2026-11-04). The FNPEG corrector fix (Marquardt-scaled LM
# damping, bound-aware step; the first runs lost range at low L/D) supersedes the first FNPEG
# final-MC runs, which are kept in superseded_fnpeg_v1/. Order: FNPEG final MC again, then the
# S5 robustness-vs-scale sets (0.1x .. 1.25x, 300 cases each) for both guidance laws, then the
# full FNPEG stability suite. campaign.ps1's own step 3 (that suite) is stopped as soon as it
# starts, so the final MC goes first; every step resumes (finished files are skipped).
param([int]$WaitPid = 0)
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root
$env:PYTHONPATH = $root
Remove-Item Env:FNPEG_CONFIG -ErrorAction SilentlyContinue
$log = Join-Path $root "results\final_mc\campaign_log.txt"
function Step($name, $argv) {
    Add-Content $log "[$(Get-Date -Format s)] START $name"
    & python @argv *>> $log
    Add-Content $log "[$(Get-Date -Format s)] END $name (exit $LASTEXITCODE)"
}
if ($WaitPid -gt 0) {
    while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) {
        if (Select-String -Path $log -Pattern "START fnpeg_suite" -Quiet -ErrorAction SilentlyContinue) {
            Get-CimInstance Win32_Process |
                Where-Object { $_.CommandLine -match "validation\.stability" -and $_.CommandLine -match "--guidance fnpeg" } |
                ForEach-Object { & taskkill /PID $_.ProcessId /T /F | Out-Null }
        }
        Start-Sleep -Seconds 20
    }
    Add-Content $log "[$(Get-Date -Format s)] campaign.ps1 finished; its fnpeg_suite step is deferred to campaign2.ps1"
}
Step "fnpeg_final_mc" @("-m", "reentry_rl.validation.final_mc", "--guidance", "fnpeg",
                        "--conditions", "ideal", "hifi", "--n", "2000", "--workers", "8")
Step "scale_sets" @("-m", "reentry_rl.validation.final_mc", "--scale-sets", "--guidance", "rl", "fnpeg",
                    "--workers", "8")
Step "fnpeg_suite" @("-m", "reentry_rl.validation.stability", "--guidance", "fnpeg",
                     "--studies", "sensitivity", "recovery", "margins", "corridor", "footprint", "combined",
                     "--outdir", "results/stability_fnpeg", "--workers", "8")
Add-Content $log "[$(Get-Date -Format s)] CAMPAIGN2 DONE"
