param(
    [string]$OutputRoot = "",
    [int]$MaxSamples = 100,
    [switch]$UpdatePaper,
    [string]$PaperSectionPath = "",
    [string]$PaperMainPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $repoRoot "artifacts\summarizer_critic_benchmark\sporc_ablation_studies\ieee_qwen36_27b_deepseek_v4_no_cap"
}

$root = (Resolve-Path $OutputRoot).Path
$planPath = Join-Path $root "ablation_plan.json"
if (-not (Test-Path -LiteralPath $planPath -PathType Leaf)) {
    throw "Ablation plan not found: $planPath"
}

$plan = Get-Content -LiteralPath $planPath -Raw | ConvertFrom-Json

foreach ($cell in $plan.cells) {
    Write-Host "Extending $($cell.cell_id) to $MaxSamples retained samples..."

    $cmd = [System.Collections.Generic.List[string]]::new()
    foreach ($part in $cell.command) {
        [void]$cmd.Add([string]$part)
    }

    for ($i = 0; $i -lt $cmd.Count; $i++) {
        if ($cmd[$i] -eq "--max-samples" -and ($i + 1) -lt $cmd.Count) {
            $cmd[$i + 1] = [string]$MaxSamples
        }
    }

    if (-not $cmd.Contains("--resume-from-checkpoint")) {
        [void]$cmd.Add("--resume-from-checkpoint")
    }

    $exe = $cmd[0]
    $cmdArgs = @()
    if ($cmd.Count -gt 1) {
        $cmdArgs = $cmd.ToArray()[1..($cmd.Count - 1)]
    }

    & $exe @cmdArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Ablation cell failed: $($cell.cell_id)"
    }
}

$reportArgs = @(
    "run",
    "python",
    "scripts/run_sporc_ablation_study.py",
    "--mode",
    "report",
    "--output-root",
    $root
)

if ($UpdatePaper) {
    if ([string]::IsNullOrWhiteSpace($PaperSectionPath) -or [string]::IsNullOrWhiteSpace($PaperMainPath)) {
        throw "Pass -PaperSectionPath and -PaperMainPath when using -UpdatePaper."
    }
    $reportArgs += @(
        "--update-paper",
        "--paper-section-path",
        $PaperSectionPath,
        "--paper-main-path",
        $PaperMainPath
    )
}

& uv @reportArgs
if ($LASTEXITCODE -ne 0) {
    throw "Ablation report generation failed."
}
