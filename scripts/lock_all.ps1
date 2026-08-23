$ErrorActionPreference = "Stop"

$AuditedUvVersion = "0.12.5"
$IrodoriRevision = "8224dafb46d0aba89209a8f905f1cb7e3299d9c1"
$IrodoriDir = "vendor/Irodori-TTS"
$ManagedIrodoriProject = "locks/Irodori-TTS.pyproject.toml"
$ManagedIrodoriLock = "locks/Irodori-TTS.uv.lock"

function Assert-NativeSuccess([string]$Label) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

function Assert-AuditedUv {
    $VersionLine = (& uv --version).Trim()
    Assert-NativeSuccess "uv --version"
    if ($VersionLine -notmatch ("^uv " + [regex]::Escape($AuditedUvVersion) + "(?:\s|$)")) {
        throw "Lock refresh requires audited uv $AuditedUvVersion, got: $VersionLine"
    }
}

function Test-WorkerExtra([string]$Worker, [string]$Extra) {
    uv sync --project "workers/$Worker" --locked --dry-run --extra $Extra
    Assert-NativeSuccess "locked $Worker/$Extra dry-run"
}

Assert-AuditedUv

uv lock
Assert-NativeSuccess "uv lock (root)"
uv sync --locked --dry-run
Assert-NativeSuccess "root locked dry-run"

Get-ChildItem workers -Directory | ForEach-Object {
    if (Test-Path (Join-Path $_.FullName "pyproject.toml")) {
        uv lock --project $_.FullName
        Assert-NativeSuccess "uv lock ($($_.Name))"
    }
}

# Validate every backend family that setup/CI is allowed to select. This makes
# the maintenance script fail immediately instead of committing a lock that only
# breaks later on a different GPU generation.
uv sync --project workers/asr --locked --dry-run
Assert-NativeSuccess "locked asr dry-run"
foreach ($Worker in @("diarization", "sense", "lfm")) {
    foreach ($Extra in @("cpu", "cu126", "cu128")) {
        Test-WorkerExtra $Worker $Extra
    }
}
foreach ($Extra in @("cpu", "cu124")) {
    Test-WorkerExtra "seed_vc" $Extra
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
        foreach ($Extra in @("cpu", "cu126", "cu128")) {
            uv sync --project $IrodoriDir --locked --dry-run --extra $Extra
            Assert-NativeSuccess "locked Irodori/$Extra dry-run"
        }
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

Write-Host "All audited uv lockfiles refreshed and backend matrices validated with uv $AuditedUvVersion."
