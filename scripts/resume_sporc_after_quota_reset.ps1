param(
    [string]$InitialRoot = "artifacts/sporc_human_eval_stage2_sweeps/full_20260321T221039Z",
    [string]$RecoveryBaseRoot = "",
    [string]$NotesPath = "artifacts/run_notes/2026-03-21_sporc_gpu_gemini_sweep.md",
    [datetime]$StartAt = [datetime]"2026-03-22T08:15:00+08:00"
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
        [int]$MaxIterations,
        [int]$SummarizerMaxToolTurns,
        [int]$CriticMaxToolTurns,
        [switch]$DisableInterjectorToolCallPlanning
    )

    $args = @(
        "run",
        "python",
        "scripts/run_sporc_human_eval_stage2_sweep.py",
        "--skip-setup",
        "--continue-on-error",
        "--episode-indices",
        $EpisodeIndices,
        "--duration-presets",
        $DurationPresets,
        "--output-root",
        $OutputRoot,
        "--orchestrator-max-iterations",
        $MaxIterations,
        "--summarizer-max-tool-turns",
        $SummarizerMaxToolTurns,
        "--critic-max-tool-turns",
        $CriticMaxToolTurns
    )
    if ($DisableInterjectorToolCallPlanning.IsPresent) {
        $args += "--disable-interjector-tool-call-planning"
    }

    $process = Start-Process `
        -FilePath $uvPath `
        -ArgumentList $args `
        -WorkingDirectory $repoRoot `
        -PassThru

    try {
        while (-not $process.HasExited) {
            Start-Sleep -Seconds 30

            $quotaMatch = Get-QuotaSignal -OutputRoot $OutputRoot
            if ($null -ne $quotaMatch) {
                Add-Note "Detected Gemini quota exhaustion while processing output root `"$OutputRoot`". Log: `"$($quotaMatch.Path)`" line $($quotaMatch.LineNumber). Stopping the stuck sweep so it can be resumed cleanly at the next quota window."
                Stop-SetupAndRunContainers
                Stop-ProcessTree -ProcessId $process.Id
                return 1
            }

            $process.Refresh()
        }
    }
    finally {
        if (-not $process.HasExited) {
            $process.Refresh()
        }
    }

    return $process.ExitCode
}

function Get-QuotaSignal {
    param([string]$OutputRoot)

    if (-not (Test-Path -Path $OutputRoot)) {
        return $null
    }

    $candidateLogs = Get-ChildItem -Path $OutputRoot -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -in @("agent_interactions.log", "main.log") }
    if (-not $candidateLogs) {
        return $null
    }

    $latestMatch = $null
    foreach ($log in $candidateLogs) {
        $match = Select-String -Path $log.FullName -Pattern "429 RESOURCE_EXHAUSTED|RESOURCE_EXHAUSTED" -CaseSensitive:$false -ErrorAction SilentlyContinue |
            Select-Object -Last 1
        if ($null -eq $match) {
            continue
        }

        $candidate = [pscustomobject]@{
            Path = $log.FullName
            LineNumber = $match.LineNumber
            Line = $match.Line
            LastWriteTime = $log.LastWriteTime
        }
        if ($null -eq $latestMatch -or $candidate.LastWriteTime -gt $latestMatch.LastWriteTime) {
            $latestMatch = $candidate
        }
    }

    return $latestMatch
}

function Stop-ProcessTree {
    param([int]$ProcessId)

    if (-not (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)) {
        return
    }

    & taskkill /PID $ProcessId /T /F | Out-Null
}

function Stop-SetupAndRunContainers {
    $containerIds = docker ps --format "{{.ID}} {{.Names}}" |
        Where-Object { $_ -match "card-framework-setup-and-run-gpu-run" } |
        ForEach-Object { ($_ -split " ", 2)[0] }

    foreach ($containerId in $containerIds) {
        docker stop $containerId | Out-Null
    }
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

if ([string]::IsNullOrWhiteSpace($RecoveryBaseRoot)) {
    $stamp = (Get-Date).ToString("yyyyMMddTHHmmssZ")
    $RecoveryBaseRoot = "artifacts/sporc_human_eval_stage2_sweeps/quota_recovery_$stamp"
}

$resolvedRecoveryBaseRoot = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $RecoveryBaseRoot))
$fiveMinuteRoot = Join-Path $resolvedRecoveryBaseRoot "remaining_5m"
$shortRoot = Join-Path $resolvedRecoveryBaseRoot "remaining_short"
$combinedRoot = Join-Path $resolvedRecoveryBaseRoot "combined_final_exports"

$delaySeconds = [int][Math]::Max(
    0,
    [Math]::Ceiling(($StartAt.ToUniversalTime() - (Get-Date).ToUniversalTime()).TotalSeconds)
)

Add-Note "Quota recovery worker armed. Waiting until $($StartAt.ToString('yyyy-MM-ddTHH:mm:sszzz')) before resuming the remaining SPoRC Gemini sweep."

if ($delaySeconds -gt 0) {
    Start-Sleep -Seconds $delaySeconds
}

Add-Note "Quota recovery worker starting the remaining 5m runs with max_iterations=3 and the standard run parameters."
$fiveMinuteExitCode = Run-Sweep `
    -OutputRoot $fiveMinuteRoot `
    -EpisodeIndices "2,3,4,5" `
    -DurationPresets "5m" `
    -MaxIterations 3 `
    -SummarizerMaxToolTurns 0 `
    -CriticMaxToolTurns 0

Add-Note "Quota recovery worker starting the remaining 30s/1m runs with max_iterations=3 and the standard run parameters."
$shortExitCode = Run-Sweep `
    -OutputRoot $shortRoot `
    -EpisodeIndices "3,4,5" `
    -DurationPresets "30s,1m" `
    -MaxIterations 3 `
    -SummarizerMaxToolTurns 0 `
    -CriticMaxToolTurns 0

Copy-FinalExports -SourceRoot $InitialRoot -DestinationRoot $combinedRoot
Copy-FinalExports -SourceRoot $fiveMinuteRoot -DestinationRoot $combinedRoot
Copy-FinalExports -SourceRoot $shortRoot -DestinationRoot $combinedRoot
Write-CombinedManifest -DestinationRoot $combinedRoot -SourceRoots @(
    $InitialRoot,
    $fiveMinuteRoot,
    $shortRoot
)

Add-Note "Quota recovery worker finished. five_minute_exit_code=$fiveMinuteExitCode; short_exit_code=$shortExitCode; combined exports root: `$resolvedRecoveryBaseRoot`."

if ($fiveMinuteExitCode -ne 0 -or $shortExitCode -ne 0) {
    exit 1
}

exit 0
