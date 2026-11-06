param(
    [switch]$LaunchDashboard
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$virtualEnvPython = Join-Path $root "venv\Scripts\python.exe"

if (Test-Path -LiteralPath $virtualEnvPython) {
    $python = $virtualEnvPython
} else {
    $pythonCommand = Get-Command python -ErrorAction Stop
    $python = $pythonCommand.Source
}

$previousDataset = $env:TEQO_DATASET
$env:TEQO_DATASET = "synthetic"
Push-Location $root

try {
    Write-Host "Using Python: $python"
    Write-Host "Running the TeqoAI synthetic graduation-demo pipeline."
    Write-Host "This regenerates synthetic inputs and processed outputs; it does not use LANL data."

    $steps = @(
        "src.generate_synthetic_data",
        "src.data_prep",
        "src.model",
        "src.risk_engine"
    )

    foreach ($step in $steps) {
        Write-Host "`n===== $step ====="
        & $python -m $step
        if ($LASTEXITCODE -ne 0) {
            throw "$step failed with exit code $LASTEXITCODE."
        }
    }

    Write-Host "`nEvaluation report: data\processed\evaluation_report.json"
    Write-Host "Scored events: data\processed\auth_events_scored.csv"
    Write-Host "Risk events: data\processed\auth_events_risk.csv"

    if ($LaunchDashboard) {
        Write-Host "`nLaunching the dashboard at http://localhost:8501/"
        & $python -m streamlit run (Join-Path $root "dashboard\app.py") `
            --server.address 127.0.0.1 --server.port 8501
        if ($LASTEXITCODE -ne 0) {
            throw "Streamlit exited with code $LASTEXITCODE."
        }
    } else {
        Write-Host "To launch the dashboard from the project root: .\venv\Scripts\python.exe -m streamlit run dashboard\app.py"
    }
} finally {
    if ($null -eq $previousDataset) {
        Remove-Item Env:\TEQO_DATASET -ErrorAction SilentlyContinue
    } else {
        $env:TEQO_DATASET = $previousDataset
    }
    Pop-Location
}
