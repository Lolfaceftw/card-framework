param(
    [string]$OutputBaseRoot = "artifacts/sporc_human_eval_stage2_sweeps/deepseek_reasoner_remaining_20260322T0915Z",
    [string]$InitialExportsRoot = "artifacts/sporc_human_eval_stage2_sweeps/full_20260321T221039Z/final_exports",
    [string]$NotesPath = "artifacts/run_notes/2026-03-21_sporc_gpu_gemini_sweep.md",
    [int]$MaxAttemptsPerRun = 5,
    [int]$RetrySleepSeconds = 20
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

$resolvedOutputBaseRoot = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $OutputBaseRoot))
$resolvedInitialExportsRoot = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $InitialExportsRoot))
$resolvedNotesPath = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $NotesPath))
$repairLogsRoot = Join-Path $resolvedOutputBaseRoot "remaining_short\repair_logs"
$repairStatusPath = Join-Path $resolvedOutputBaseRoot "repair_missing_outputs_status.json"
$remainingShortRunSummaryPath = Join-Path $resolvedOutputBaseRoot "remaining_short\run_summary.json"
$combinedFinalExportsRoot = Join-Path $resolvedOutputBaseRoot "combined_final_exports"
$combinedManifestPath = Join-Path $resolvedOutputBaseRoot "combined_manifest.json"

New-Item -ItemType Directory -Force -Path $repairLogsRoot | Out-Null
New-Item -ItemType Directory -Force -Path $combinedFinalExportsRoot | Out-Null

function Add-Note {
    param([string]$Message)
    $timestamp = (Get-Date).ToString("yyyy-MM-ddTHH:mm:sszzz")
    $content = @"

### $timestamp

- $Message
"@
    Add-Content -Path $resolvedNotesPath -Value $content -Encoding UTF8
}

function To-ContainerPath {
    param([string]$PathValue)
    $fullPath = [System.IO.Path]::GetFullPath($PathValue)
    if (-not $fullPath.StartsWith($repoRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Path '$fullPath' is not under repo root '$repoRoot'."
    }
    $relative = $fullPath.Substring($repoRoot.Length).TrimStart('\')
    if ([string]::IsNullOrWhiteSpace($relative)) {
        return "/workspace/card-framework"
    }
    return "/workspace/card-framework/" + ($relative -replace "\\", "/")
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

function Read-JsonFile {
    param([string]$PathValue)
    if (-not (Test-Path -Path $PathValue)) {
        return $null
    }
    $raw = Get-Content -Path $PathValue -Raw -Encoding UTF8
    if ([string]::IsNullOrWhiteSpace($raw)) {
        return $null
    }
    return $raw | ConvertFrom-Json -Depth 100
}

function Write-JsonFile {
    param(
        [string]$PathValue,
        [object]$Payload
    )
    $dir = Split-Path -Parent $PathValue
    if (-not [string]::IsNullOrWhiteSpace($dir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
    }
    $json = $Payload | ConvertTo-Json -Depth 100
    Set-Content -Path $PathValue -Value $json -Encoding UTF8
}

function Set-RepairStatus {
    param(
        [string]$Status,
        [string]$CurrentRun = "",
        [int]$Attempt = 0,
        [string]$LastError = ""
    )
    $payload = [ordered]@{
        updated_at = (Get-Date).ToString("o")
        status = $Status
        current_run = $CurrentRun
        attempt = $Attempt
        completed_outputs = Get-UniqueCompletedOutputCount
        last_error = $LastError
    }
    Write-JsonFile -PathValue $repairStatusPath -Payload $payload
}

function Invoke-CommandWithTee {
    param(
        [string[]]$Command,
        [string]$LogPath
    )
    $dir = Split-Path -Parent $LogPath
    if (-not (Test-Path -Path $dir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
    }
    Set-Content -Path $LogPath -Value ('$ ' + ($Command -join ' ') + [Environment]::NewLine + [Environment]::NewLine) -Encoding UTF8
    $commandName = $Command[0]
    $commandArgs = @()
    if ($Command.Count -gt 1) {
        $commandArgs = $Command[1..($Command.Count - 1)]
    }
    & $commandName @commandArgs 2>&1 | Tee-Object -FilePath $LogPath -Append
    return $LASTEXITCODE
}

function Get-RunDefinitions {
    return @(
        [pscustomobject]@{
            EpisodeIndex = 3
            Title = "079 The United State of St Kitts and Nevis ft Nekirah Nicholls"
            DurationLabel = "1m"
            TargetSeconds = 60
            RunRoot = Join-Path $resolvedOutputBaseRoot "remaining_short\runs\03_079_the_united_state_of_st_kitts_and_nevis_ft_nekirah_nicholls\1m"
            SummaryExportName = "03 - 079 The United State of St Kitts and Nevis ft Nekirah Nicholls - 1m - summary.xml"
            AudioExportName = "03 - 079 The United State of St Kitts and Nevis ft Nekirah Nicholls - 1m - voice_cloned_interjected.wav"
        }
        [pscustomobject]@{
            EpisodeIndex = 4
            Title = "Ep 6: Polite & Passive Aggression"
            DurationLabel = "30s"
            TargetSeconds = 30
            RunRoot = Join-Path $resolvedOutputBaseRoot "remaining_short\runs\04_ep_6_polite_passive_aggression\30s"
            SummaryExportName = "04 - Ep 6 - Polite & Passive Aggression - 30s - summary.xml"
            AudioExportName = "04 - Ep 6 - Polite & Passive Aggression - 30s - voice_cloned_interjected.wav"
        }
        [pscustomobject]@{
            EpisodeIndex = 4
            Title = "Ep 6: Polite & Passive Aggression"
            DurationLabel = "1m"
            TargetSeconds = 60
            RunRoot = Join-Path $resolvedOutputBaseRoot "remaining_short\runs\04_ep_6_polite_passive_aggression\1m"
            SummaryExportName = "04 - Ep 6 - Polite & Passive Aggression - 1m - summary.xml"
            AudioExportName = "04 - Ep 6 - Polite & Passive Aggression - 1m - voice_cloned_interjected.wav"
        }
        [pscustomobject]@{
            EpisodeIndex = 5
            Title = "Video Game Logic Episode 195: Take Two Takedown"
            DurationLabel = "1m"
            TargetSeconds = 60
            RunRoot = Join-Path $resolvedOutputBaseRoot "remaining_short\runs\05_video_game_logic_episode_195_take_two_takedown\1m"
            SummaryExportName = "05 - Video Game Logic Episode 195 - Take Two Takedown - 1m - summary.xml"
            AudioExportName = "05 - Video Game Logic Episode 195 - Take Two Takedown - 1m - voice_cloned_interjected.wav"
        }
    )
}

function Build-RunCommand {
    param([pscustomobject]$RunDef)
    $transcriptPath = Join-Path $RunDef.RunRoot "transcript.json"
    return @(
        "docker",
        "compose",
        "--profile",
        "gpu",
        "run",
        "--rm",
        "setup-and-run-gpu",
        "--no-menu",
        "--output-root",
        (To-ContainerPath -PathValue $RunDef.RunRoot),
        "--override",
        "pipeline.start_stage=stage-2",
        "--override",
        ("transcript_path=" + (To-ContainerPath -PathValue $transcriptPath)),
        "--override",
        "selection.embedding=disabled",
        "--override",
        "selection.llm=deepseek_reasoner",
        "--override",
        "selection.stage_llm.summarizer=deepseek_reasoner",
        "--override",
        "selection.stage_llm.critic=deepseek_reasoner",
        "--override",
        "selection.stage_llm.interjector=deepseek_reasoner",
        "--override",
        "profiles.llm.deepseek_reasoner.model=deepseek-reasoner",
        "--override",
        ("orchestrator.target_seconds=" + [string]$RunDef.TargetSeconds),
        "--override",
        "audio.device=cuda",
        "--override",
        "audio.voice_clone.device=cuda",
        "--override",
        "audio.voice_clone.enabled=true",
        "--override",
        "audio.interjector.enabled=true",
        "--override",
        "logging.print_to_terminal=true",
        "--override",
        "logging.summarizer_critic_print_to_terminal=true",
        "--override",
        "orchestrator.timeouts.summarizer=900",
        "--override",
        "orchestrator.timeouts.critic=900",
        "--override",
        "orchestrator.timeouts.retrieval=300",
        "--override",
        "orchestrator.timeouts.index=300",
        "--override",
        "orchestrator.max_iterations=3"
    )
}

function Export-RunArtifacts {
    param([pscustomobject]$RunDef)
    $sourceSummaryPath = Join-Path $RunDef.RunRoot "summary.xml"
    $sourceAudioPath = Join-Path $RunDef.RunRoot "audio_stage\interjector\voice_cloned_interjected.wav"
    $finalExportsRoot = Join-Path $resolvedOutputBaseRoot "remaining_short\final_exports"
    New-Item -ItemType Directory -Force -Path $finalExportsRoot | Out-Null
    $destSummaryPath = Join-Path $finalExportsRoot $RunDef.SummaryExportName
    $destAudioPath = Join-Path $finalExportsRoot $RunDef.AudioExportName
    Copy-Item -Path $sourceSummaryPath -Destination $destSummaryPath -Force
    Copy-Item -Path $sourceAudioPath -Destination $destAudioPath -Force
    return [pscustomobject]@{
        summary_xml = $destSummaryPath
        final_audio = $destAudioPath
    }
}

function Update-RunSummary {
    param(
        [pscustomobject]$RunDef,
        [string]$LogPath,
        [int]$ReturnCode,
        [string]$Status,
        [object]$ExportPaths
    )
    $payload = Read-JsonFile -PathValue $remainingShortRunSummaryPath
    if ($null -eq $payload) {
        return
    }
    foreach ($run in @($payload.runs)) {
        if (
            ([int]$run.episode_index -eq [int]$RunDef.EpisodeIndex) -and
            ([string]$run.duration_preset -eq [string]$RunDef.DurationLabel)
        ) {
            $run.status = $Status
            $run.log_path = $LogPath
            $run.return_code = $ReturnCode
            if ($Status -eq "completed") {
                $run.artifacts = [ordered]@{
                    summary_xml_path = (Join-Path $RunDef.RunRoot "summary.xml")
                    transcript_path = (Join-Path $RunDef.RunRoot "transcript.json")
                    speaker_manifest_path = (Join-Path $RunDef.RunRoot "speaker_samples\manifest.json")
                    voice_clone_manifest_path = (Join-Path $RunDef.RunRoot "audio_stage\voice_clone\manifest.json")
                    interjector_manifest_path = (Join-Path $RunDef.RunRoot "audio_stage\interjector\interjector_manifest.json")
                    final_audio_path = (Join-Path $RunDef.RunRoot "audio_stage\interjector\voice_cloned_interjected.wav")
                }
                if ($null -ne $ExportPaths) {
                    $run.human_readable_exports = [ordered]@{
                        summary_xml = [string]$ExportPaths.summary_xml
                        final_audio = [string]$ExportPaths.final_audio
                    }
                }
            }
            break
        }
    }
    Write-JsonFile -PathValue $remainingShortRunSummaryPath -Payload $payload
}

function Merge-CombinedExports {
    $sourceRoots = @(
        $resolvedInitialExportsRoot,
        (Join-Path $resolvedOutputBaseRoot "remaining_5m\final_exports"),
        (Join-Path $resolvedOutputBaseRoot "remaining_short\final_exports")
    )
    New-Item -ItemType Directory -Force -Path $combinedFinalExportsRoot | Out-Null
    foreach ($root in $sourceRoots) {
        if (-not (Test-Path -Path $root)) {
            continue
        }
        Get-ChildItem -Path $root -File -ErrorAction SilentlyContinue | ForEach-Object {
            Copy-Item -Path $_.FullName -Destination (Join-Path $combinedFinalExportsRoot $_.Name) -Force
        }
    }
}

function Write-CombinedManifest {
    $audioFiles = Get-ChildItem -Path $combinedFinalExportsRoot -File -Filter "* - voice_cloned_interjected.wav" -ErrorAction SilentlyContinue |
        Sort-Object Name
    $records = @()
    foreach ($file in $audioFiles) {
        $provider = if (
            $file.Name -like "01 -*" -or
            $file.Name -like "02 - Spirit Adrift* - 30s*" -or
            $file.Name -like "02 - Spirit Adrift* - 1m*"
        ) {
            "gemini-3.1-pro-preview"
        }
        else {
            "deepseek-reasoner"
        }
        $records += [ordered]@{
            file_name = $file.Name
            provider_model = $provider
            path = $file.FullName
        }
    }
    $payload = [ordered]@{
        generated_at = (Get-Date).ToString("o")
        total_final_audio = @($audioFiles).Count
        combined_final_exports_root = $combinedFinalExportsRoot
        files = $records
    }
    Write-JsonFile -PathValue $combinedManifestPath -Payload $payload
}

Add-Note "DeepSeek missing-output repair wrapper started. Reusing the existing failed run roots and their loop-memory artifacts. Researcher-set parameters remain unchanged, including DeepSeek Reasoner and orchestrator.max_iterations=3."
Set-RepairStatus -Status "running" -CurrentRun "" -Attempt 0 -LastError ""

$runDefinitions = Get-RunDefinitions
foreach ($runDef in $runDefinitions) {
    $finalAudioPath = Join-Path $resolvedOutputBaseRoot ("remaining_short\final_exports\" + $runDef.AudioExportName)
    if (Test-Path -Path $finalAudioPath) {
        Add-Note ("Skipping already-completed DeepSeek repair target `"" + $runDef.Title + "`" at `"" + $runDef.DurationLabel + "`".")
        continue
    }

    $succeeded = $false
    for ($attempt = 1; $attempt -le $MaxAttemptsPerRun; $attempt++) {
        $runLabel = "$($runDef.EpisodeIndex)-$($runDef.DurationLabel)"
        $logPath = Join-Path $repairLogsRoot ("ep{0:D2}_{1}_attempt{2:D2}.log" -f [int]$runDef.EpisodeIndex, $runDef.DurationLabel, $attempt)
        Add-Note ("DeepSeek repair attempt " + $attempt + "/" + $MaxAttemptsPerRun + " for `"" + $runDef.Title + "`" at `"" + $runDef.DurationLabel + "`" using the existing run root `"" + $runDef.RunRoot + "`".")
        Set-RepairStatus -Status "running" -CurrentRun $runLabel -Attempt $attempt -LastError ""
        $command = Build-RunCommand -RunDef $runDef
        $returnCode = Invoke-CommandWithTee -Command $command -LogPath $logPath

        $sourceSummaryPath = Join-Path $runDef.RunRoot "summary.xml"
        $sourceAudioPath = Join-Path $runDef.RunRoot "audio_stage\interjector\voice_cloned_interjected.wav"
        if (($returnCode -eq 0) -and (Test-Path -Path $sourceSummaryPath) -and (Test-Path -Path $sourceAudioPath)) {
            $exports = Export-RunArtifacts -RunDef $runDef
            Update-RunSummary -RunDef $runDef -LogPath $logPath -ReturnCode $returnCode -Status "completed" -ExportPaths $exports
            Add-Note ("DeepSeek repair succeeded for `"" + $runDef.Title + "`" at `"" + $runDef.DurationLabel + "`" on attempt " + $attempt + ". Exported summary and final audio into remaining_short/final_exports.")
            $succeeded = $true
            break
        }

        Update-RunSummary -RunDef $runDef -LogPath $logPath -ReturnCode $returnCode -Status "failed" -ExportPaths $null
        $tail = ""
        if (Test-Path -Path $logPath) {
            $tail = (Get-Content -Path $logPath -Tail 12 -ErrorAction SilentlyContinue | Where-Object { $_.Trim().Length -gt 0 } | Select-Object -Last 1)
        }
        Add-Note ("DeepSeek repair attempt failed for `"" + $runDef.Title + "`" at `"" + $runDef.DurationLabel + "`" with exit code " + $returnCode + ". Log: `"" + $logPath + "`". Tail: `"" + $tail + "`".")
        if ($attempt -lt $MaxAttemptsPerRun) {
            Start-Sleep -Seconds $RetrySleepSeconds
        }
    }

    if (-not $succeeded) {
        $message = "DeepSeek repair exhausted " + $MaxAttemptsPerRun + " attempts for $($runDef.Title) at $($runDef.DurationLabel)."
        Set-RepairStatus -Status "failed" -CurrentRun "$($runDef.EpisodeIndex)-$($runDef.DurationLabel)" -Attempt $MaxAttemptsPerRun -LastError $message
        Add-Note $message
        throw $message
    }
}

Merge-CombinedExports
Write-CombinedManifest
$completedOutputs = Get-UniqueCompletedOutputCount
Set-RepairStatus -Status "completed" -CurrentRun "" -Attempt 0 -LastError ""
Add-Note ("DeepSeek missing-output repair wrapper finished with total_outputs=" + $completedOutputs + "/15. Combined exports refreshed under `"" + $combinedFinalExportsRoot + "`".")

