# Follow-up to campaign2.ps1 (2026-11-04): the FNPEG phase-0 fix (phase 0 only while a pull-out
# above v_pull is predicted; shallow entries had flown lift-up to the handover) supersedes the
# FNPEG S5 scale sets (kept in scale/superseded_phase0/) and the FNPEG stability suite started by
# campaign2 (stopped after 6 min). The final MC is re-run only with -FinalMC: a phase-0 scan of
# its 2 x 2000 cases decides (identical switch times = identical trajectories). Every step
# resumes (finished files are skipped).
param([switch]$FinalMC)
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
if ($FinalMC) {
    Step "fnpeg_final_mc" @("-m", "reentry_rl.validation.final_mc", "--guidance", "fnpeg",
                            "--conditions", "ideal", "hifi", "--n", "2000", "--workers", "8")
}
Step "fnpeg_scale_sets" @("-m", "reentry_rl.validation.final_mc", "--scale-sets", "--guidance", "fnpeg",
                          "--workers", "8")
Step "fnpeg_suite" @("-m", "reentry_rl.validation.stability", "--guidance", "fnpeg",
                     "--studies", "sensitivity", "recovery", "margins", "corridor", "footprint", "combined",
                     "--outdir", "results/stability_fnpeg", "--workers", "8")
Add-Content $log "[$(Get-Date -Format s)] CAMPAIGN3 DONE"
