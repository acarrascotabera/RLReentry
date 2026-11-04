# Paper evaluation campaign (2026-11-04): final Monte Carlo, then the new margin conditions of
# the RL policy, then the full stability suite of the FNPEG baseline. Each step resumes
# (finished files are skipped), so the script can be relaunched after an interruption.
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
Step "final_mc" @("-m", "reentry_rl.validation.final_mc", "--guidance", "rl", "fnpeg",
                  "--conditions", "ideal", "hifi", "--n", "2000", "--workers", "8")
Step "rl_new_margins" @("-m", "reentry_rl.validation.stability", "--studies", "margins",
                        "--outdir", "results/stability_v26", "--workers", "8")
Step "fnpeg_suite" @("-m", "reentry_rl.validation.stability", "--guidance", "fnpeg",
                     "--studies", "sensitivity", "recovery", "margins", "corridor", "footprint", "combined",
                     "--outdir", "results/stability_fnpeg", "--workers", "8")
Add-Content $log "[$(Get-Date -Format s)] CAMPAIGN DONE"
