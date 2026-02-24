param(
    [string]$Version = "8.19.3"
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..\\..")
$pidFile = Join-Path (Join-Path $root ".tools") ("elasticsearch-" + $Version + ".pid")

if (-not (Test-Path $pidFile)) {
    Write-Output "No pid file found: $pidFile"
    exit 0
}

$pidValue = [int](Get-Content $pidFile | Select-Object -First 1)
if ($pidValue -le 0) {
    Remove-Item $pidFile -Force
    Write-Output "Invalid pid file removed."
    exit 0
}

try {
    Stop-Process -Id $pidValue -Force
    Write-Output ("Stopped elasticsearch pid=" + $pidValue)
} catch {
    Write-Output ("Process not running: " + $pidValue)
}

Remove-Item $pidFile -Force
