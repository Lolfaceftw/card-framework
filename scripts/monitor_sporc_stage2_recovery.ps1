param(
    [string]$SweepParent = "artifacts/sporc_human_eval_stage2_sweeps",
    [string]$NotesPath = "artifacts/run_notes/2026-03-21_sporc_gpu_gemini_sweep.md",
    [int]$PollSeconds = 120
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

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

function Get-ContainerSnapshot {
    $lines = docker ps --format "{{.Names}}|{{.Status}}" 2>$null
    if (-not $lines) {
        return [pscustomobject]@{
            Summary = "containers=none"
            Names = @()
        }
    }

    $rows = @($lines | Where-Object { $_ -match "card-framework-setup-and-run-gpu-run" })
    if (-not $rows) {
        return [pscustomobject]@{
            Summary = "containers=none"
            Names = @()
        }
    }

    $names = @()
    $summaryParts = @()
    foreach ($row in $rows) {
        $parts = $row -split "\|", 2
        $name = $parts[0].Trim()
        $status = if ($parts.Count -gt 1) { $parts[1].Trim() } else { "unknown" }
        $names += $name
        $summaryParts += "$name ($status)"
    }

    return [pscustomobject]@{
        Summary = "containers=" + ($summaryParts -join ", ")
        Names = $names
    }
}

function Get-RunSnapshot {
    param([string]$ParentPath)

    $resolvedParent = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $ParentPath))
    if (-not (Test-Path -Path $resolvedParent)) {
        return "runs=0/15"
    }

    $completedExports = @(
        Get-ChildItem -Path $resolvedParent -Recurse -File -ErrorAction SilentlyContinue |
            Where-Object {
                $_.Name -like "* - voice_cloned_interjected.wav" -and
                $_.DirectoryName -like "*\final_exports"
            }
    )
    $completed = @($completedExports | Select-Object -ExpandProperty Name -Unique).Count

    $summaryFiles = Get-ChildItem -Path $resolvedParent -Recurse -Filter run_summary.json -File -ErrorAction SilentlyContinue
    if (-not $summaryFiles) {
        return "runs=$completed/15"
    }
    $pendingDetails = @()

    foreach ($summaryFile in $summaryFiles) {
        try {
            $payload = Get-Content -Path $summaryFile.FullName -Raw -Encoding UTF8 | ConvertFrom-Json -Depth 8
        }
        catch {
            continue
        }

        foreach ($run in @($payload.runs)) {
            $status = [string]$run.status
            if ($status -eq "pending" -or $status -eq "prepared" -or $status -eq "failed") {
                $episode = [string]$run.episode_title
                $preset = [string]$run.duration_preset
                $pendingDetails += "${status}:$episode/$preset"
            }
        }
    }

    $summary = "runs=$completed/15"
    if ($pendingDetails.Count -gt 0) {
        $summary += " pending=" + (($pendingDetails | Sort-Object | Select-Object -First 3) -join "; ")
    }
    return $summary
}

function Get-RecoveryWorkerSnapshot {
    $processes = Get-CimInstance Win32_Process -Filter "Name = 'powershell.exe'" |
        Where-Object { $_.CommandLine -like "*resume_sporc_after_quota_reset.ps1*" }
    if (-not $processes) {
        return "recovery_worker=none"
    }

    $snapshots = @()
    foreach ($process in $processes) {
        $snapshots += "pid=$($process.ProcessId)"
    }
    return "recovery_worker=" + ($snapshots -join ",")
}

function Get-LatestQuotaSignal {
    param([string]$ParentPath)

    $resolvedParent = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $ParentPath))
    if (-not (Test-Path -Path $resolvedParent)) {
        return $null
    }

    $logs = Get-ChildItem -Path $resolvedParent -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -eq "agent_interactions.log" }
    if (-not $logs) {
        return $null
    }

    $latest = $null
    foreach ($log in $logs) {
        $match = Select-String -Path $log.FullName -Pattern "RESOURCE_EXHAUSTED" -CaseSensitive:$false -ErrorAction SilentlyContinue |
            Select-Object -Last 1
        if ($null -eq $match) {
            continue
        }

        if ($null -eq $latest -or $log.LastWriteTime -gt $latest.LastWriteTime) {
            $latest = [pscustomobject]@{
                Path = $log.FullName
                LastWriteTime = $log.LastWriteTime
                Line = $match.Line.Trim()
            }
        }
    }

    return $latest
}

Add-Note "Monitor worker armed for the SPoRC Gemini GPU sweep. Poll interval: ${PollSeconds}s."

$lastState = ""
$lastQuotaPath = ""

while ($true) {
    $containerSnapshot = Get-ContainerSnapshot
    $gpuSnapshot = Get-GpuSnapshot
    $runSnapshot = Get-RunSnapshot -ParentPath $SweepParent
    $workerSnapshot = Get-RecoveryWorkerSnapshot

    $state = "$runSnapshot | $($containerSnapshot.Summary) | $workerSnapshot | $gpuSnapshot"
    if ($state -ne $lastState) {
        Add-Note "Monitor update: $state."
        $lastState = $state
    }

    $quotaSignal = Get-LatestQuotaSignal -ParentPath $SweepParent
    if ($null -ne $quotaSignal -and $quotaSignal.Path -ne $lastQuotaPath) {
        Add-Note "Latest quota signal still present at `"$($quotaSignal.Path)`": $($quotaSignal.Line)"
        $lastQuotaPath = $quotaSignal.Path
    }

    if ($runSnapshot -like "runs=15/15*") {
        Add-Note "Monitor worker observed all 15 outputs completed and is exiting."
        break
    }

    Start-Sleep -Seconds $PollSeconds
}
