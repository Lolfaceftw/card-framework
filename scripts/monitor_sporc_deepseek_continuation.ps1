param(
    [string]$InitialRoot = "artifacts/sporc_human_eval_stage2_sweeps/full_20260321T221039Z",
    [string]$OutputBaseRoot,
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

function Get-UniqueCompletedOutputCount {
    $roots = @(
        [System.IO.Path]::GetFullPath((Join-Path $repoRoot $InitialRoot)),
        [System.IO.Path]::GetFullPath((Join-Path $repoRoot $OutputBaseRoot))
    ) | Select-Object -Unique

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
        Where-Object { $_.CommandLine -like "*run_sporc_remaining_with_deepseek.ps1*" }
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

function Get-RunSummarySnapshot {
    $resolvedRoot = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $OutputBaseRoot))
    if (-not (Test-Path -Path $resolvedRoot)) {
        return "deepseek_runs=0"
    }

    $summaryFiles = Get-ChildItem -Path $resolvedRoot -Recurse -Filter run_summary.json -File -ErrorAction SilentlyContinue
    if (-not $summaryFiles) {
        return "deepseek_runs=0"
    }

    $completed = 0
    $pending = 0
    $failed = 0
    $pendingLabels = @()
    foreach ($summaryFile in $summaryFiles) {
        try {
            $payload = Get-Content -Path $summaryFile.FullName -Raw -Encoding UTF8 | ConvertFrom-Json -Depth 8
        }
        catch {
            continue
        }
        foreach ($run in @($payload.runs)) {
            $status = [string]$run.status
            switch ($status) {
                "completed" { $completed += 1 }
                "failed" {
                    $failed += 1
                    $pendingLabels += "failed:$($run.episode_title)/$($run.duration_preset)"
                }
                default {
                    $pending += 1
                    $pendingLabels += "${status}:$($run.episode_title)/$($run.duration_preset)"
                }
            }
        }
    }

    $summary = "deepseek_runs completed=$completed pending=$pending failed=$failed"
    if ($pendingLabels.Count -gt 0) {
        $summary += " current=" + (($pendingLabels | Select-Object -First 2) -join "; ")
    }
    return $summary
}

function Get-LatestLogSnapshot {
    $resolvedRoot = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $OutputBaseRoot))
    if (-not (Test-Path -Path $resolvedRoot)) {
        return "latest_log=n/a"
    }

    $latestLog = Get-ChildItem -Path $resolvedRoot -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Extension -eq ".log" } |
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

if ([string]::IsNullOrWhiteSpace($OutputBaseRoot)) {
    throw "OutputBaseRoot is required."
}

Add-Note "DeepSeek continuation monitor armed for `"$OutputBaseRoot`" with a ${PollSeconds}s interval."

while ($true) {
    $completed = Get-UniqueCompletedOutputCount
    $wrapper = Get-WrapperSnapshot
    $containers = Get-ContainerSnapshot
    $gpu = Get-GpuSnapshot
    $runs = Get-RunSummarySnapshot
    $latestLog = Get-LatestLogSnapshot

    Add-Note "DeepSeek monitor heartbeat: total_outputs=$completed/15; $wrapper; $containers; $gpu; $runs; $latestLog"

    if ($completed -ge 15) {
        Add-Note "DeepSeek continuation monitor observed all 15 final outputs and is exiting."
        break
    }

    Start-Sleep -Seconds $PollSeconds
}
