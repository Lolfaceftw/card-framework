param(
    [string]$InitialRoot = "artifacts/sporc_human_eval_stage2_sweeps/full_20260321T221039Z",
    [string]$OutputBaseRoot = "",
    [string]$NotesPath = "artifacts/run_notes/2026-03-21_sporc_gpu_gemini_sweep.md"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot
$uvPath = (Get-Command uv -ErrorAction Stop).Source

function Add-Note {
    param([string]$Message)
    $resolvedNotesPath = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $NotesPath))
    $timestamp = (Get-Date).ToString("yyyy-MM-ddTHH:mm:sszzz")
    $content = @"

### $timestamp

- $Message
"@
    Add-Content -Path $resolvedNotesPath -Value $content -Encoding UTF8
}

function Run-Sweep {
    param(
        [string]$OutputRoot,
        [string]$EpisodeIndices,
        [string]$DurationPresets,
        [switch]$SkipSetup
    )

    $args = @(
        "run",
        "python",
        "scripts/run_sporc_human_eval_stage2_sweep.py",
        "--continue-on-error",
        "--llm-profile",
        "deepseek_reasoner",
        "--llm-model",
        "deepseek-reasoner",
        "--episode-indices",
        $EpisodeIndices,
        "--duration-presets",
        $DurationPresets,
        "--output-root",
        $OutputRoot,
        "--orchestrator-max-iterations",
        "3"
    )
    if ($SkipSetup.IsPresent) {
        $args += "--skip-setup"
    }

    $process = Start-Process `
        -FilePath $uvPath `
        -ArgumentList $args `
        -WorkingDirectory $repoRoot `
        -PassThru `
        -Wait

    return $process.ExitCode
}

function Copy-FinalExports {
    param(
        [string]$SourceRoot,
        [string]$DestinationRoot
    )

    $sourceExports = [System.IO.Path]::GetFullPath((Join-Path $repoRoot (Join-Path $SourceRoot "final_exports")))
    if (-not (Test-Path -Path $sourceExports)) {
        return
    }
    New-Item -ItemType Directory -Force -Path $DestinationRoot | Out-Null
    Get-ChildItem -Path $sourceExports -File | ForEach-Object {
        Copy-Item -Path $_.FullName -Destination (Join-Path $DestinationRoot $_.Name) -Force
    }
}

function Write-CombinedManifest {
    param(
        [string]$DestinationRoot,
        [string[]]$SourceRoots
    )

    $files = Get-ChildItem -Path $DestinationRoot -File | Sort-Object Name | ForEach-Object {
        [ordered]@{
            name = $_.Name
            path = $_.FullName
            length = $_.Length
            provider = if ($_.Name -match "^01 - " -or $_.Name -match "^02 - .* - (30s|1m) - ") {
                "gemini-3.1-pro-preview"
            } else {
                "deepseek-reasoner"
            }
        }
    }
    $payload = [ordered]@{
        generated_at = (Get-Date).ToString("o")
        source_roots = $SourceRoots
        file_count = @($files).Count
        files = $files
    }
    $manifestPath = Join-Path $DestinationRoot "combined_manifest.json"
    $payload | ConvertTo-Json -Depth 6 | Set-Content -Path $manifestPath -Encoding UTF8
}

if ([string]::IsNullOrWhiteSpace($OutputBaseRoot)) {
    $stamp = (Get-Date).ToString("yyyyMMddTHHmmssZ")
    $OutputBaseRoot = "artifacts/sporc_human_eval_stage2_sweeps/deepseek_remaining_$stamp"
}

$resolvedOutputBaseRoot = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $OutputBaseRoot))
$fiveMinuteRoot = Join-Path $resolvedOutputBaseRoot "remaining_5m"
$shortRoot = Join-Path $resolvedOutputBaseRoot "remaining_short"
$combinedRoot = Join-Path $resolvedOutputBaseRoot "combined_final_exports"

Add-Note "Starting DeepSeek Reasoner continuation for the remaining 10 SPoRC outputs. Existing Gemini outputs under `"$InitialRoot`" remain preserved and untouched."

$fiveMinuteExitCode = Run-Sweep `
    -OutputRoot $fiveMinuteRoot `
    -EpisodeIndices "2,3,4,5" `
    -DurationPresets "5m"

$shortExitCode = Run-Sweep `
    -OutputRoot $shortRoot `
    -EpisodeIndices "3,4,5" `
    -DurationPresets "30s,1m" `
    -SkipSetup

Copy-FinalExports -SourceRoot $InitialRoot -DestinationRoot $combinedRoot
Copy-FinalExports -SourceRoot $fiveMinuteRoot -DestinationRoot $combinedRoot
Copy-FinalExports -SourceRoot $shortRoot -DestinationRoot $combinedRoot
Write-CombinedManifest -DestinationRoot $combinedRoot -SourceRoots @(
    $InitialRoot,
    $fiveMinuteRoot,
    $shortRoot
)

Add-Note "DeepSeek Reasoner continuation finished. five_minute_exit_code=$fiveMinuteExitCode; short_exit_code=$shortExitCode; combined exports root: `"$resolvedOutputBaseRoot`"."

if ($fiveMinuteExitCode -ne 0 -or $shortExitCode -ne 0) {
    exit 1
}

exit 0
