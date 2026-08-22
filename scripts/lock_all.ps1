$ErrorActionPreference = "Stop"

$IrodoriRevision = "8224dafb46d0aba89209a8f905f1cb7e3299d9c1"
$IrodoriDir = "vendor/Irodori-TTS"
$ManagedIrodoriProject = "locks/Irodori-TTS.pyproject.toml"
$ManagedIrodoriLock = "locks/Irodori-TTS.uv.lock"

function Assert-NativeSuccess([string]$Label) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

uv lock
Assert-NativeSuccess "uv lock (root)"

Get-ChildItem workers -Directory | ForEach-Object {
    if (Test-Path (Join-Path $_.FullName "pyproject.toml")) {
        uv lock --project $_.FullName
        Assert-NativeSuccess "uv lock ($($_.Name))"
    }
}

if (Test-Path (Join-Path $IrodoriDir "pyproject.toml")) {
    if (-not (Test-Path $ManagedIrodoriProject)) {
        throw "Managed Irodori project overlay is missing: $ManagedIrodoriProject"
    }

    $HeadRevision = (& git -C $IrodoriDir rev-parse HEAD).Trim()
    Assert-NativeSuccess "git rev-parse (Irodori)"
    if ($HeadRevision -ne $IrodoriRevision) {
        throw "Irodori checkout is not at the audited revision: $HeadRevision"
    }

    $Status = (& git -C $IrodoriDir status --porcelain) -join "`n"
    Assert-NativeSuccess "git status (Irodori)"
    if ($Status) {
        throw "Irodori checkout has local changes; refusing to generate a managed lock."
    }

    New-Item -ItemType Directory -Force -Path "locks" | Out-Null
    $VendorProject = Join-Path $IrodoriDir "pyproject.toml"
    $VendorLock = Join-Path $IrodoriDir "uv.lock"
    $ProjectBackup = [System.IO.File]::ReadAllBytes((Resolve-Path $VendorProject))
    $HadLock = Test-Path $VendorLock
    $LockBackup = if ($HadLock) { [System.IO.File]::ReadAllBytes((Resolve-Path $VendorLock)) } else { $null }

    try {
        Copy-Item -Force $ManagedIrodoriProject $VendorProject
        uv lock --project $IrodoriDir
        Assert-NativeSuccess "uv lock (Irodori managed overlay)"
        Copy-Item -Force $VendorLock $ManagedIrodoriLock
    }
    finally {
        [System.IO.File]::WriteAllBytes((Join-Path (Get-Location) $VendorProject), $ProjectBackup)
        if ($HadLock) {
            [System.IO.File]::WriteAllBytes((Join-Path (Get-Location) $VendorLock), $LockBackup)
        }
        else {
            Remove-Item -Force -ErrorAction SilentlyContinue $VendorLock
        }
    }
}
else {
    Write-Host "Irodori vendor checkout is absent; managed Irodori lock was left unchanged."
}

Write-Host "All audited uv lockfiles refreshed."
