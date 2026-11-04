# Follow-up to campaign.ps1: waits for it to finish, then runs the S5 robustness-vs-scale sets
# (0.1x .. 1.25x, 300 cases each) for the policy and the FNPEG baseline through the same harness.
param([int]$WaitPid = 0)
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root
$env:PYTHONPATH = $root
Remove-Item Env:FNPEG_CONFIG -ErrorAction SilentlyContinue
$log = Join-Path $root "results\final_mc\campaign_log.txt"
if ($WaitPid -gt 0) {
    while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 60 }
}
Add-Content $log "[$(Get-Date -Format s)] START scale_sets"
& python -m reentry_rl.validation.final_mc --scale-sets --guidance rl fnpeg --workers 8 *>> $log
Add-Content $log "[$(Get-Date -Format s)] END scale_sets (exit $LASTEXITCODE)"
