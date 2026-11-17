$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$pythonExe = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    & python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Cannot create Python virtual environment.' }
}
$nodeDirectory = Join-Path $PSScriptRoot '.tools\node-v22.14.0-win-x64'
if (-not (Test-Path -LiteralPath (Join-Path $nodeDirectory 'node.exe'))) {
    New-Item -ItemType Directory -Force -Path .tools | Out-Null
    $archivePath = Join-Path $PSScriptRoot '.tools\node-v22.14.0-win-x64.zip'
    Invoke-WebRequest -UseBasicParsing -Uri 'https://nodejs.org/dist/v22.14.0/node-v22.14.0-win-x64.zip' -OutFile $archivePath
    if ((Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne '55b639295920b219bb2acbcfa00f90393a2789095b7323f79475c9f34795f217') {
        throw 'Node.js SHA256 mismatch.'
    }
    Expand-Archive -LiteralPath $archivePath -DestinationPath .tools -Force
}
$env:PATH = $nodeDirectory + ';' + $env:PATH
& $pythonExe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed.' }
& $pythonExe -m pip install -r backend/requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Python dependencies installation failed.' }
& $pythonExe -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency conflict detected.' }
& $pythonExe scripts/setup_models.py
if ($LASTEXITCODE -ne 0) { throw 'Official model verification/download failed.' }
Push-Location frontend
try {
    & npm.cmd ci
    if ($LASTEXITCODE -ne 0) { throw 'Frontend dependencies installation failed.' }
    & npm.cmd run build
    if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
} finally { Pop-Location }
Write-Host 'Setup complete. Start: python run.py'
