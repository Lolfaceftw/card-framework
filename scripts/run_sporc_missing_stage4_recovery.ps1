param(
    [string]$OutputBaseRoot = "artifacts/sporc_human_eval_stage2_sweeps/deepseek_reasoner_remaining_20260322T0915Z",
    [string]$GeminiExportsRoot = "artifacts/sporc_human_eval_stage2_sweeps/full_20260321T221039Z/final_exports",
    [string]$FiveMinuteExportsRoot = "artifacts/sporc_human_eval_stage2_sweeps/deepseek_reasoner_remaining_20260322T0915Z/remaining_5m/final_exports",
    [string]$NotesPath = "artifacts/run_notes/2026-03-21_sporc_gpu_gemini_sweep.md"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

$resolvedOutputBaseRoot = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $OutputBaseRoot))
$resolvedGeminiExportsRoot = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $GeminiExportsRoot))
$resolvedFiveMinuteExportsRoot = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $FiveMinuteExportsRoot))
$resolvedNotesPath = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $NotesPath))
$shortFinalExportsRoot = Join-Path $resolvedOutputBaseRoot "remaining_short\final_exports"
$recoveryLogsRoot = Join-Path $resolvedOutputBaseRoot "remaining_short\recovery_stage4_logs"
$combinedFinalExportsRoot = Join-Path $resolvedOutputBaseRoot "combined_final_exports"
$combinedManifestPath = Join-Path $resolvedOutputBaseRoot "combined_manifest.json"
$statusPath = Join-Path $resolvedOutputBaseRoot "missing_stage4_recovery_status.json"

New-Item -ItemType Directory -Force -Path $shortFinalExportsRoot | Out-Null
New-Item -ItemType Directory -Force -Path $recoveryLogsRoot | Out-Null
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

function Read-JsonFile {
    param([string]$PathValue)
    if (-not (Test-Path -Path $PathValue)) {
        return $null
    }
    $raw = Get-Content -Path $PathValue -Raw -Encoding UTF8
    if ([string]::IsNullOrWhiteSpace($raw)) {
        return $null
    }
    return $raw | ConvertFrom-Json
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

function Set-ObjectProperty {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Object,
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [object]$Value
    )
    $existing = $Object.PSObject.Properties[$Name]
    if ($null -ne $existing) {
        $existing.Value = $Value
        return
    }
    $Object | Add-Member -NotePropertyName $Name -NotePropertyValue $Value -Force
}

function Get-UniqueCompletedOutputCount {
    $roots = @(
        $resolvedGeminiExportsRoot,
        $resolvedFiveMinuteExportsRoot,
        $shortFinalExportsRoot
    ) | Select-Object -Unique
    $audioNames = @()
    foreach ($root in $roots) {
        if (-not (Test-Path -Path $root)) {
            continue
        }
        $audioNames += @(
            Get-ChildItem -Path $root -File -Filter "*voice_cloned_interjected.wav" -ErrorAction SilentlyContinue |
                Select-Object -ExpandProperty Name
        )
    }
    return @($audioNames | Sort-Object -Unique).Count
}

function Get-MemorySnapshot {
    $vmmem = Get-Process -ErrorAction SilentlyContinue |
        Where-Object { $_.ProcessName -eq "vmmemWSL" } |
        Select-Object -First 1
    $os = Get-CimInstance Win32_OperatingSystem
    $snapshot = [ordered]@{
        total_visible_gb = [math]::Round($os.TotalVisibleMemorySize / 1MB, 2)
        free_physical_gb = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
        free_virtual_gb = [math]::Round($os.FreeVirtualMemory / 1MB, 2)
        vmmem_working_set_gb = if ($null -ne $vmmem) { [math]::Round($vmmem.WorkingSet64 / 1GB, 2) } else { $null }
        vmmem_private_gb = if ($null -ne $vmmem) { [math]::Round($vmmem.PrivateMemorySize64 / 1GB, 2) } else { $null }
    }
    return $snapshot
}

function Format-MemorySnapshot {
    $snapshot = Get-MemorySnapshot
    $vmmemWorkingSet = if ($null -ne $snapshot.vmmem_working_set_gb) { [string]$snapshot.vmmem_working_set_gb } else { "n/a" }
    $vmmemPrivate = if ($null -ne $snapshot.vmmem_private_gb) { [string]$snapshot.vmmem_private_gb } else { "n/a" }
    return (
        "memory: free_physical_gb=" + $snapshot.free_physical_gb +
        ", free_virtual_gb=" + $snapshot.free_virtual_gb +
        ", vmmem_working_set_gb=" + $vmmemWorkingSet +
        ", vmmem_private_gb=" + $vmmemPrivate
    )
}

function Set-RecoveryStatus {
    param(
        [string]$Status,
        [string]$CurrentRun = "",
        [string]$Stage = "",
        [string]$LastError = "",
        [string]$LogPath = ""
    )
    $payload = [ordered]@{
        updated_at = (Get-Date).ToString("o")
        status = $Status
        current_run = $CurrentRun
        stage = $Stage
        completed_outputs = Get-UniqueCompletedOutputCount
        last_error = $LastError
        log_path = $LogPath
        memory_snapshot = Get-MemorySnapshot
    }
    Write-JsonFile -PathValue $statusPath -Payload $payload
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
    $previousErrorActionPreference = $ErrorActionPreference
    $global:ErrorActionPreference = "Continue"
    try {
        & $commandName @commandArgs 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $LogPath -Append | Out-Null
        return $LASTEXITCODE
    }
    finally {
        $global:ErrorActionPreference = $previousErrorActionPreference
    }
}

function Get-RunDefinitions {
    return @(
        [pscustomobject]@{
            RunKey = "03_1m"
            EpisodeIndex = 3
            DurationLabel = "1m"
            TitleForNote = "079 The United State of St Kitts and Nevis ft Nekirah Nicholls"
            OutputRoot = Join-Path $resolvedOutputBaseRoot "remaining_short\runs\03_079_the_united_state_of_st_kitts_and_nevis_ft_nekirah_nicholls\1m"
            SummaryExportName = "03 - 079 The United State of St Kitts and Nevis ft Nekirah Nicholls - 1m - summary.xml"
            AudioExportName = "03 - 079 The United State of St Kitts and Nevis ft Nekirah Nicholls - 1m - voice_cloned_interjected.wav"
        }
        [pscustomobject]@{
            RunKey = "04_30s"
            EpisodeIndex = 4
            DurationLabel = "30s"
            TitleForNote = "Ep 6 - Polite & Passive Aggression"
            OutputRoot = Join-Path $resolvedOutputBaseRoot "remaining_short\runs\04_ep_6_polite_passive_aggression\30s"
            SummaryExportName = "04 - Ep 6 - Polite & Passive Aggression - 30s - summary.xml"
            AudioExportName = "04 - Ep 6 - Polite & Passive Aggression - 30s - voice_cloned_interjected.wav"
        }
        [pscustomobject]@{
            RunKey = "04_1m"
            EpisodeIndex = 4
            DurationLabel = "1m"
            TitleForNote = "Ep 6 - Polite & Passive Aggression"
            OutputRoot = Join-Path $resolvedOutputBaseRoot "remaining_short\runs\04_ep_6_polite_passive_aggression\1m"
            SummaryExportName = "04 - Ep 6 - Polite & Passive Aggression - 1m - summary.xml"
            AudioExportName = "04 - Ep 6 - Polite & Passive Aggression - 1m - voice_cloned_interjected.wav"
        }
        [pscustomobject]@{
            RunKey = "05_1m"
            EpisodeIndex = 5
            DurationLabel = "1m"
            TitleForNote = "Video Game Logic Episode 195 - Take Two Takedown"
            OutputRoot = Join-Path $resolvedOutputBaseRoot "remaining_short\runs\05_video_game_logic_episode_195_take_two_takedown\1m"
            SummaryExportName = "05 - Video Game Logic Episode 195 - Take Two Takedown - 1m - summary.xml"
            AudioExportName = "05 - Video Game Logic Episode 195 - Take Two Takedown - 1m - voice_cloned_interjected.wav"
        }
    )
}

function Build-Stage3RecoveryCommand {
    param([pscustomobject]$RunDef)
    $containerRunRoot = To-ContainerPath -PathValue $RunDef.OutputRoot
    return @(
        "docker",
        "compose",
        "--profile",
        "gpu",
        "run",
        "--rm",
        "--entrypoint",
        "bash",
        "setup-and-run-gpu",
        "-lc",
        ("cd /workspace/card-framework && uv run python scripts/recover_sporc_live_draft_stage3.py --run-root '" + $containerRunRoot + "'")
    )
}

function Build-Stage4Command {
    param([pscustomobject]$RunDef)
    $containerRunRoot = To-ContainerPath -PathValue $RunDef.OutputRoot
    $containerSummaryPath = To-ContainerPath -PathValue (Join-Path $RunDef.OutputRoot "summary.xml")
    $containerVoiceCloneManifestPath = To-ContainerPath -PathValue (Join-Path $RunDef.OutputRoot "audio_stage\voice_clone\manifest.json")
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
        $containerRunRoot,
        "--override",
        "pipeline.start_stage=stage-4",
        "--override",
        ("pipeline.final_summary_path=" + $containerSummaryPath),
        "--override",
        ("pipeline.voice_clone_manifest_path=" + $containerVoiceCloneManifestPath),
        "--override",
        "selection.embedding=disabled",
        "--override",
        "selection.llm=deepseek_reasoner",
        "--override",
        "selection.stage_llm.interjector=deepseek_reasoner",
        "--override",
        "profiles.llm.deepseek_reasoner.model=deepseek-reasoner",
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
        "logging.summarizer_critic_print_to_terminal=true"
    )
}

function Get-RunReview {
    param(
        [string]$SummaryPath,
        [string]$InterjectorManifestPath
    )
    $summaryXml = Get-Content -Path $SummaryPath -Raw -Encoding UTF8
    $wrapped = [xml]("<summary>" + $summaryXml + "</summary>")
    $turnNodes = @($wrapped.summary.ChildNodes | Where-Object { $_.NodeType -eq "Element" })
    $previewLines = @()
    foreach ($node in ($turnNodes | Select-Object -First 3)) {
        $previewText = ($node.InnerText -replace "\s+", " ").Trim()
        if ($previewText.Length -gt 120) {
            $previewText = $previewText.Substring(0, 117) + "..."
        }
        $previewLines += ($node.Name + ": " + $previewText)
    }
    $interjectorManifest = Read-JsonFile -PathValue $InterjectorManifestPath
    $turnText = ($turnNodes | ForEach-Object { $_.InnerText }) -join " "
    $wordCount = @(($turnText -split "\s+") | Where-Object { $_ -match "\S" }).Count
    return [ordered]@{
        summary_xml_path = $SummaryPath
        interjector_manifest_path = $InterjectorManifestPath
        turn_count = $turnNodes.Count
        word_count = $wordCount
        speakers = @($turnNodes | ForEach-Object { $_.Name })
        preview_lines = $previewLines
        interjection_artifact_count = [int]($interjectorManifest.artifact_count)
        concerns = @(
            "Recovered from the saved live draft after the DeepSeek stage-2 run hit max_iterations=3 without critic convergence."
        )
    }
}

function Export-RunArtifacts {
    param([pscustomobject]$RunDef)
    $summarySource = Join-Path $RunDef.OutputRoot "summary.xml"
    $audioSource = Join-Path $RunDef.OutputRoot "audio_stage\interjector\voice_cloned_interjected.wav"
    $summaryDest = Join-Path $shortFinalExportsRoot $RunDef.SummaryExportName
    $audioDest = Join-Path $shortFinalExportsRoot $RunDef.AudioExportName
    Copy-Item -Path $summarySource -Destination $summaryDest -Force
    Copy-Item -Path $audioSource -Destination $audioDest -Force
    return [ordered]@{
        summary_xml = $summaryDest
        final_audio = $audioDest
    }
}

function Update-RunSummary {
    param(
        [pscustomobject]$RunDef,
        [string]$Stage4LogPath,
        [object]$ExportPaths
    )
    $runSummaryPath = Join-Path $resolvedOutputBaseRoot "remaining_short\run_summary.json"
    $payload = Read-JsonFile -PathValue $runSummaryPath
    if ($null -eq $payload) {
        return
    }
    foreach ($run in @($payload.runs)) {
        if (
            ([int]$run.episode_index -eq [int]$RunDef.EpisodeIndex) -and
            ([string]$run.duration_preset -eq [string]$RunDef.DurationLabel)
        ) {
            $summaryPath = Join-Path $RunDef.OutputRoot "summary.xml"
            $voiceCloneManifestPath = Join-Path $RunDef.OutputRoot "audio_stage\voice_clone\manifest.json"
            $interjectorManifestPath = Join-Path $RunDef.OutputRoot "audio_stage\interjector\interjector_manifest.json"
            $finalAudioPath = Join-Path $RunDef.OutputRoot "audio_stage\interjector\voice_cloned_interjected.wav"
            $run.status = "completed"
            $run.return_code = 0
            $run.log_path = $Stage4LogPath
            Set-ObjectProperty -Object $run -Name "recovery_method" -Value "live_draft_stage3_plus_stage4_recovery"
            Set-ObjectProperty -Object $run -Name "artifacts" -Value ([ordered]@{
                summary_xml_path = $summaryPath
                transcript_path = [string]$run.transcript_path
                speaker_manifest_path = Join-Path $RunDef.OutputRoot "speaker_samples\manifest.json"
                voice_clone_manifest_path = $voiceCloneManifestPath
                interjector_manifest_path = $interjectorManifestPath
                final_audio_path = $finalAudioPath
            })
            Set-ObjectProperty -Object $run -Name "human_readable_exports" -Value $ExportPaths
            Set-ObjectProperty -Object $run -Name "llm_output_review" -Value (Get-RunReview -SummaryPath $summaryPath -InterjectorManifestPath $interjectorManifestPath)
        }
    }
    Write-JsonFile -PathValue $runSummaryPath -Payload $payload
}

function Sync-CombinedExports {
    $sources = @(
        $resolvedGeminiExportsRoot,
        $resolvedFiveMinuteExportsRoot,
        $shortFinalExportsRoot
    ) | Select-Object -Unique
    if (Test-Path -Path $combinedFinalExportsRoot) {
        Get-ChildItem -Path $combinedFinalExportsRoot -File -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
    }
    $copied = @()
    foreach ($source in $sources) {
        if (-not (Test-Path -Path $source)) {
            continue
        }
        foreach ($file in Get-ChildItem -Path $source -File -ErrorAction SilentlyContinue) {
            $destination = Join-Path $combinedFinalExportsRoot $file.Name
            Copy-Item -Path $file.FullName -Destination $destination -Force
            $copied += [ordered]@{
                name = $file.Name
                source_path = $file.FullName
                combined_path = $destination
            }
        }
    }
    $manifest = [ordered]@{
        generated_at = (Get-Date).ToString("o")
        total_files = @($copied).Count
        files = $copied
    }
    Write-JsonFile -PathValue $combinedManifestPath -Payload $manifest
}

function Ensure-ExpectedArtifacts {
    param([pscustomobject]$RunDef)
    $required = @(
        (Join-Path $RunDef.OutputRoot "summary.xml"),
        (Join-Path $RunDef.OutputRoot "audio_stage\voice_clone\manifest.json"),
        (Join-Path $RunDef.OutputRoot "audio_stage\voice_clone\voice_cloned.wav"),
        (Join-Path $RunDef.OutputRoot "audio_stage\interjector\interjector_manifest.json"),
        (Join-Path $RunDef.OutputRoot "audio_stage\interjector\voice_cloned_interjected.wav")
    )
    foreach ($path in $required) {
        if (-not (Test-Path -Path $path)) {
            throw "Expected artifact missing after recovery: $path"
        }
    }
}

Add-Note "Starting missing-output recovery using saved live-draft stage-3 artifacts plus DeepSeek Reasoner stage-4 only. Research parameters stay pinned, including max_iterations=3, and no summarizer/critic loop files are being changed."
Add-Note ("Initial " + (Format-MemorySnapshot) + ".")
Set-RecoveryStatus -Status "starting" -Stage "bootstrap"

$runDefs = Get-RunDefinitions
foreach ($runDef in $runDefs) {
    $stage3LogPath = Join-Path $recoveryLogsRoot ($runDef.RunKey + "_stage3.log")
    $stage4LogPath = Join-Path $recoveryLogsRoot ($runDef.RunKey + "_stage4.log")
    $expectedExportAudio = Join-Path $shortFinalExportsRoot $runDef.AudioExportName
    if (Test-Path -Path $expectedExportAudio) {
        $existingSummaryPath = Join-Path $runDef.OutputRoot "summary.xml"
        $existingInterjectorManifestPath = Join-Path $runDef.OutputRoot "audio_stage\interjector\interjector_manifest.json"
        if ((Test-Path -Path $existingSummaryPath) -and (Test-Path -Path $existingInterjectorManifestPath)) {
            $existingExports = [ordered]@{
                summary_xml = (Join-Path $shortFinalExportsRoot $runDef.SummaryExportName)
                final_audio = $expectedExportAudio
            }
            Update-RunSummary -RunDef $runDef -Stage4LogPath $stage4LogPath -ExportPaths $existingExports
        }
        Add-Note ("Skipping " + $runDef.RunKey + " because the final export already exists at " + $expectedExportAudio + ".")
        continue
    }

    Set-RecoveryStatus -Status "running" -CurrentRun $runDef.RunKey -Stage "stage3_recovery" -LogPath $stage3LogPath
    Add-Note ("Rebuilding stage-3 artifacts from saved live-draft state for " + $runDef.RunKey + " (" + $runDef.TitleForNote + ", " + $runDef.DurationLabel + "). " + (Format-MemorySnapshot) + ".")
    $stage3ReturnCode = Invoke-CommandWithTee -Command (Build-Stage3RecoveryCommand -RunDef $runDef) -LogPath $stage3LogPath
    if ($stage3ReturnCode -ne 0) {
        $message = "Stage-3 live-draft recovery failed for " + $runDef.RunKey + " with exit code " + $stage3ReturnCode + "."
        Set-RecoveryStatus -Status "failed" -CurrentRun $runDef.RunKey -Stage "stage3_recovery" -LastError $message -LogPath $stage3LogPath
        Add-Note $message
        exit $stage3ReturnCode
    }

    $interjectorDir = Join-Path $runDef.OutputRoot "audio_stage\interjector"
    if (Test-Path -Path $interjectorDir) {
        Remove-Item -Path $interjectorDir -Recurse -Force -ErrorAction SilentlyContinue
    }

    Set-RecoveryStatus -Status "running" -CurrentRun $runDef.RunKey -Stage "stage4_recovery" -LogPath $stage4LogPath
    Add-Note ("Running DeepSeek Reasoner stage-4 interjector recovery for " + $runDef.RunKey + ". " + (Format-MemorySnapshot) + ".")
    $stage4ReturnCode = Invoke-CommandWithTee -Command (Build-Stage4Command -RunDef $runDef) -LogPath $stage4LogPath
    if ($stage4ReturnCode -ne 0) {
        $message = "Stage-4 recovery failed for " + $runDef.RunKey + " with exit code " + $stage4ReturnCode + "."
        Set-RecoveryStatus -Status "failed" -CurrentRun $runDef.RunKey -Stage "stage4_recovery" -LastError $message -LogPath $stage4LogPath
        Add-Note $message
        exit $stage4ReturnCode
    }

    Ensure-ExpectedArtifacts -RunDef $runDef
    $exports = Export-RunArtifacts -RunDef $runDef
    Update-RunSummary -RunDef $runDef -Stage4LogPath $stage4LogPath -ExportPaths $exports
    Set-RecoveryStatus -Status "completed_run" -CurrentRun $runDef.RunKey -Stage "stage4_recovery" -LogPath $stage4LogPath
    Add-Note ("Recovered " + $runDef.RunKey + " to a final audio export at " + $exports.final_audio + ". " + (Format-MemorySnapshot) + ".")
}

Sync-CombinedExports
$totalCompleted = Get-UniqueCompletedOutputCount
if ($totalCompleted -lt 15) {
    $message = "Recovery wrapper finished without reaching 15 outputs. Current unique final audio count: " + $totalCompleted + "/15."
    Set-RecoveryStatus -Status "failed" -Stage "verification" -LastError $message
    Add-Note $message
    exit 1
}

Set-RecoveryStatus -Status "completed" -Stage "verification"
Add-Note ("Recovery wrapper verified " + $totalCompleted + "/15 final audio outputs and refreshed combined_final_exports at " + $combinedFinalExportsRoot + ".")
