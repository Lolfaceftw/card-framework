param(
    [int]$NumSamples = 2,
    [string]$ManifestPath = "artifacts\diarization_datasets\ami\manifest_test_2.json",
    [string]$DataRoot = "artifacts\diarization_datasets\ami",
    [string]$OutputDir = "artifacts\diarization_benchmark_ground_truth",
    [switch]$SkipBuild
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

$run = "speakerfix-ami-nemo-$(Get-Date -Format yyyyMMdd-HHmmss)"
$profileDir = "artifacts\resource_profiles"
New-Item -ItemType Directory -Force $profileDir | Out-Null

if (-not $SkipBuild) {
    docker compose --profile gpu build setup-and-run-gpu
}

docker compose --profile gpu run --rm --entrypoint bash setup-and-run-gpu -lc `
"uv run python -m card_framework.benchmark.diarization prepare-manifest \
  --output $ManifestPath \
  --data-root $DataRoot \
  --subset test \
  --num-samples $NumSamples"

$container = "cf-$run"
$bench = Start-Process docker `
    -ArgumentList @(
        "compose", "--profile", "gpu", "run", "--name", $container, "--rm",
        "--entrypoint", "bash", "setup-and-run-gpu", "-lc",
        "uv run --with pyannote.metrics python -m card_framework.benchmark.diarization execute --manifest $ManifestPath --output-dir $OutputDir --providers nemo --device cuda --max-samples $NumSamples --collar 0.0"
    ) `
    -WindowStyle Hidden `
    -PassThru `
    -RedirectStandardOutput "$profileDir\$run.stdout.txt" `
    -RedirectStandardError "$profileDir\$run.stderr.txt"

$telemetry = @()
while (-not $bench.HasExited) {
    $gpu = nvidia-smi --query-gpu=timestamp,name,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
    $ram = Get-CimInstance Win32_OperatingSystem | Select-Object TotalVisibleMemorySize,FreePhysicalMemory
    $stats = docker stats --no-stream --format "{{json .}}" $container 2>$null

    $telemetry += [pscustomobject]@{
        timestamp = (Get-Date).ToString("o")
        gpu = $gpu
        total_ram_mb = [math]::Round($ram.TotalVisibleMemorySize / 1024, 2)
        free_ram_mb = [math]::Round($ram.FreePhysicalMemory / 1024, 2)
        docker_stats = $stats
    }

    Start-Sleep -Seconds 5
}

$bench.WaitForExit()
$telemetry | ConvertTo-Json -Depth 6 | Set-Content "$profileDir\$run.telemetry.json"
Write-Host "exit_code=$($bench.ExitCode)"
Write-Host "stdout=$profileDir\$run.stdout.txt"
Write-Host "stderr=$profileDir\$run.stderr.txt"
Write-Host "telemetry=$profileDir\$run.telemetry.json"

if ($bench.ExitCode -ne 0) {
    throw "Diarization benchmark failed with exit code $($bench.ExitCode)."
}
