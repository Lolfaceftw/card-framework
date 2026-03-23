param(
    [string]$InitialRoot = "artifacts/sporc_human_eval_stage2_sweeps/full_20260321T221039Z",
    [string]$RecoveryBaseRoot = "artifacts/sporc_human_eval_stage2_sweeps/quota_recovery_20260323T000500Z",
    [string]$NotesPath = "artifacts/run_notes/2026-03-21_sporc_gpu_gemini_sweep.md",
    [datetime]$RecoveryStartAt = [datetime]"2026-03-23T08:05:00+08:00",
    [int]$PollSeconds = 60
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
        [System.IO.Path]::GetFullPath((Join-Path $repoRoot $RecoveryBaseRoot))
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

function Get-WorkerProcess {
    param([string]$ScriptName)

    return Get-CimInstance Win32_Process -Filter "Name = 'powershell.exe'" |
        Where-Object { $_.CommandLine -like "*$ScriptName*" }
}

function Start-DetachedPowerShellFile {
    param(
        [string]$FilePath,
        [string[]]$Arguments
    )

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = "powershell.exe"
    $psi.Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$FilePath`" " + ($Arguments -join " ")
    $psi.WorkingDirectory = $repoRoot
    $psi.UseShellExecute = $true
    $psi.CreateNoWindow = $true
    return [System.Diagnostics.Process]::Start($psi)
}

function Ensure-MonitorWorker {
    $monitor = @(Get-WorkerProcess -ScriptName "monitor_sporc_stage2_recovery.ps1")
    if ($monitor.Count -gt 0) {
        return [int]$monitor[0].ProcessId
    }

    $started = Start-DetachedPowerShellFile `
        -FilePath (Join-Path $repoRoot "scripts\monitor_sporc_stage2_recovery.ps1") `
        -Arguments @("-PollSeconds", "$PollSeconds")
    Add-Note "Supervisor restarted the local monitor worker as PID $($started.Id)."
    return [int]$started.Id
}

function Ensure-RecoveryWorker {
    $recovery = @(Get-WorkerProcess -ScriptName "resume_sporc_after_quota_reset.ps1")
    if ($recovery.Count -gt 0) {
        return [int]$recovery[0].ProcessId
    }

    $started = Start-DetachedPowerShellFile `
        -FilePath (Join-Path $repoRoot "scripts\resume_sporc_after_quota_reset.ps1") `
        -Arguments @(
            "-StartAt",
            $RecoveryStartAt.ToString("yyyy-MM-ddTHH:mm:sszzz"),
            "-RecoveryBaseRoot",
            $RecoveryBaseRoot
        )
    if ((Get-Date) -lt $RecoveryStartAt) {
        Add-Note "Supervisor restarted the waiting recovery worker as PID $($started.Id) for the unchanged recovery window $($RecoveryStartAt.ToString('yyyy-MM-ddTHH:mm:sszzz'))."
    }
    else {
        Add-Note "Supervisor restarted the recovery worker immediately as PID $($started.Id) because the scheduled recovery window had already opened and outputs were still incomplete."
    }
    return [int]$started.Id
}

Add-Note "Supervisor worker armed for the SPoRC Gemini GPU sweep. It will keep the recovery and monitor workers alive until all 15 unique final outputs exist."

$lastState = ""

while ($true) {
    $completed = Get-UniqueCompletedOutputCount
    if ($completed -ge 15) {
        Add-Note "Supervisor observed all 15 unique final outputs and is exiting."
        break
    }

    $monitorPid = Ensure-MonitorWorker
    $recoveryPid = Ensure-RecoveryWorker
    $state = "runs=$completed/15; monitor_pid=$monitorPid; recovery_pid=$recoveryPid"
    if ($state -ne $lastState) {
        Add-Note "Supervisor heartbeat: $state."
        $lastState = $state
    }

    Start-Sleep -Seconds $PollSeconds
}
