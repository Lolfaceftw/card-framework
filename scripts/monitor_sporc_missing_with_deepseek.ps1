param(
    [string]$OutputBaseRoot = "artifacts/sporc_human_eval_stage2_sweeps/deepseek_reasoner_remaining_20260322T0915Z",
    [string]$InitialExportsRoot = "artifacts/sporc_human_eval_stage2_sweeps/full_20260321T221039Z/final_exports",
    [string]$NotesPath = "artifacts/run_notes/2026-03-21_sporc_gpu_gemini_sweep.md",
    [int]$PollSeconds = 120
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

$resolvedOutputBaseRoot = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $OutputBaseRoot))
$resolvedInitialExportsRoot = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $InitialExportsRoot))
$resolvedNotesPath = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $NotesPath))
$repairStatusPath = Join-Path $resolvedOutputBaseRoot "repair_missing_outputs_status.json"
$repairLogsRoot = Join-Path $resolvedOutputBaseRoot "remaining_short\repair_logs"

function Add-Note {
    param([string]$Message)
    $timestamp = (Get-Date).ToString("yyyy-MM-ddTHH:mm:sszzz")
    $content = @"

### $timestamp

- $Message
"@
    Add-Content -Path $resolvedNotesPath -Value $content -Encoding UTF8
}

function Get-UniqueCompletedOutputCount {
    $roots = @($resolvedInitialExportsRoot, $resolvedOutputBaseRoot) | Select-Object -Unique
    $audioNames = @()
    foreach ($root in $roots) {
        if (-not (Test-Path -Path $root)) {
            continue
        }
        $audioNames += @(
            Get-ChildItem -Path $root -Recurse -File -ErrorAction SilentlyContinue |
                Where-Object {
                    $_.Name -like "* - voice_cloned_interjected.wav" -and
                    $_.DirectoryName -like "*\final_exports"
                } |
                Select-Object -ExpandProperty Name
        )
    }
    return @($audioNames | Sort-Object -Unique).Count
}

function Get-WrapperSnapshot {
    $processes = Get-CimInstance Win32_Process -Filter "Name = 'powershell.exe'" |
        Where-Object { $_.CommandLine -like "*run_sporc_missing_with_deepseek.ps1*" }
    if (-not $processes) {
        return "wrapper=none"
    }
    return "wrapper_pid=" + (($processes | Select-Object -ExpandProperty ProcessId) -join ",")
}

function Get-ContainerSnapshot {
    $lines = docker ps --format "{{.Names}}|{{.Status}}" 2>$null
    if (-not $lines) {
        return "containers=none"
    }
    $rows = @($lines | Where-Object { $_ -match "card-framework-setup-and-run-gpu-run" })
    if (-not $rows) {
        return "containers=none"
    }
    return "containers=" + (
        ($rows | ForEach-Object {
            $parts = $_ -split "\|", 2
            "$($parts[0]) ($($parts[1]))"
        }) -join ", "
    )
}

function Get-GpuSnapshot {
    $raw = & nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>$null
    if (-not $raw) {
        return "gpu=n/a"
    }
    $line = ($raw | Select-Object -First 1).Trim()
    $parts = $line -split "\s*,\s*"
    if ($parts.Count -lt 3) {
        return "gpu=n/a"
    }
    return "gpu_mem=$($parts[0])MiB/$($parts[1])MiB util=$($parts[2])%"
}

function Get-RepairStatusSnapshot {
    if (-not (Test-Path -Path $repairStatusPath)) {
        return [pscustomobject]@{
            Status = "missing"
            Summary = "repair_status=missing"
        }
    }
    try {
        $payload = Get-Content -Path $repairStatusPath -Raw -Encoding UTF8 | ConvertFrom-Json -Depth 20
    }
    catch {
        return [pscustomobject]@{
            Status = "invalid"
            Summary = "repair_status=invalid"
        }
    }
    $summary = "repair_status=$($payload.status)"
    if (-not [string]::IsNullOrWhiteSpace([string]$payload.current_run)) {
        $summary += " current=$($payload.current_run)"
    }
    if ([int]$payload.attempt -gt 0) {
        $summary += " attempt=$($payload.attempt)"
    }
    if (-not [string]::IsNullOrWhiteSpace([string]$payload.last_error)) {
        $summary += " error=$($payload.last_error)"
    }
    return [pscustomobject]@{
        Status = [string]$payload.status
        Summary = $summary
    }
}

function Get-LatestLogSnapshot {
    if (-not (Test-Path -Path $repairLogsRoot)) {
        return "latest_log=n/a"
    }
    $latestLog = Get-ChildItem -Path $repairLogsRoot -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $latestLog) {
        return "latest_log=n/a"
    }
    $tail = Get-Content -Path $latestLog.FullName -Tail 5 -ErrorAction SilentlyContinue |
        Where-Object { $_.Trim().Length -gt 0 } |
        Select-Object -Last 1
    if ([string]::IsNullOrWhiteSpace($tail)) {
        return "latest_log=$($latestLog.Name)"
    }
    $compact = ($tail -replace "\s+", " ").Trim()
    if ($compact.Length -gt 220) {
        $compact = $compact.Substring(0, 217) + "..."
    }
    return "latest_log=$($latestLog.Name): $compact"
}

Add-Note ("DeepSeek missing-output monitor armed for `"" + $OutputBaseRoot + "`" with a " + $PollSeconds + "s interval.")

while ($true) {
    $completed = Get-UniqueCompletedOutputCount
    $wrapper = Get-WrapperSnapshot
    $containers = Get-ContainerSnapshot
    $gpu = Get-GpuSnapshot
    $repair = Get-RepairStatusSnapshot
    $latestLog = Get-LatestLogSnapshot

    Add-Note ("DeepSeek repair monitor heartbeat: total_outputs=" + $completed + "/15; " + $wrapper + "; " + $containers + "; " + $gpu + "; " + $repair.Summary + "; " + $latestLog)

    if ($completed -ge 15) {
        Add-Note "DeepSeek repair monitor observed all 15 final outputs and is exiting."
        break
    }
    if ($repair.Status -eq "failed") {
        Add-Note "DeepSeek repair monitor observed a hard repair failure and is exiting."
        exit 1
    }
    if (($repair.Status -eq "completed") -and ($completed -lt 15)) {
        Add-Note "DeepSeek repair monitor observed the wrapper finish without reaching 15 outputs and is exiting."
        exit 1
    }

    Start-Sleep -Seconds $PollSeconds
}
