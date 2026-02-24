param(
    [string]$Version = "8.19.3",
    [int]$Port = 9200,
    [int]$StartupTimeoutSec = 900,
    [int]$PollIntervalSec = 2
)

$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..\\..")
$tools = Join-Path $root ".tools"
$esDir = Join-Path $tools ("elasticsearch-" + $Version)
$zipPath = Join-Path $tools ("elasticsearch-" + $Version + "-windows-x86_64.zip")
$pidFile = Join-Path $tools ("elasticsearch-" + $Version + ".pid")
$cfgFile = Join-Path $esDir "config\\elasticsearch.yml"
$logDir = Join-Path $tools "es-logs"
$url = "http://127.0.0.1:$Port"

if (-not (Test-Path $tools)) {
    New-Item -ItemType Directory -Path $tools | Out-Null
}

function Test-ElasticUp {
    try {
        $resp = Invoke-WebRequest -Uri $url -TimeoutSec 2
        return ($resp.StatusCode -eq 200)
    } catch {
        return $false
    }
}

if (Test-ElasticUp) {
    Write-Output "Elasticsearch already up at $url"
    exit 0
}

if (-not (Test-Path $esDir)) {
    if (-not (Test-Path $zipPath)) {
        $downloadUrl = "https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-$Version-windows-x86_64.zip"
        Write-Output "Downloading $downloadUrl"
        $ProgressPreference = "SilentlyContinue"
        Invoke-WebRequest -Uri $downloadUrl -OutFile $zipPath
    }
    Write-Output "Extracting $zipPath"
    Expand-Archive -Path $zipPath -DestinationPath $tools -Force
}

if (-not (Select-String -Path $cfgFile -Pattern "# codex-local overrides" -Quiet)) {
Add-Content -Path $cfgFile -Value @"

# codex-local overrides
discovery.type: single-node
xpack.security.enabled: false
http.host: 127.0.0.1
http.port: $Port
path.data: ../../.tools/es-data
path.logs: ../../.tools/es-logs
"@
}

$bat = Resolve-Path (Join-Path $esDir "bin\\elasticsearch.bat")
$proc = Start-Process -FilePath $bat -WorkingDirectory $esDir -PassThru -WindowStyle Hidden
Set-Content -Path $pidFile -Value $proc.Id
Write-Output ("Started elasticsearch pid=" + $proc.Id)

if ($PollIntervalSec -lt 1) {
    $PollIntervalSec = 1
}
if ($StartupTimeoutSec -lt $PollIntervalSec) {
    $StartupTimeoutSec = $PollIntervalSec
}

$maxPolls = [int][Math]::Ceiling($StartupTimeoutSec / $PollIntervalSec)
for ($i = 0; $i -lt $maxPolls; $i++) {
    if (Test-ElasticUp) {
        Write-Output "Elasticsearch is up at $url"
        exit 0
    }
    Start-Sleep -Seconds $PollIntervalSec
}

if (Test-ElasticUp) {
    Write-Output "Elasticsearch is up at $url"
    exit 0
}

$isRunning = $false
try {
    $null = Get-Process -Id $proc.Id -ErrorAction Stop
    $isRunning = $true
} catch {
    $isRunning = $false
}

if ($isRunning) {
    Write-Warning ("Elasticsearch process " + $proc.Id + " is still running, but health check did not pass within timeout.")
    if (Test-Path $logDir) {
        Write-Warning ("Check logs under: " + $logDir)
    }
} else {
    Write-Warning ("Elasticsearch process exited early. pid=" + $proc.Id)
    if (Test-Path $logDir) {
        Write-Warning ("Check logs under: " + $logDir)
    }
}

Write-Error ("Elasticsearch failed to start within timeout (" + $StartupTimeoutSec + "s).")
