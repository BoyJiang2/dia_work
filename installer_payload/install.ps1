$ErrorActionPreference = "Stop"

$appName = "DIA_System"
$displayName = "Digital Image Analysis System"
$sourceDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$sourceExe = Join-Path $sourceDir "DIA_System.exe"
$installDir = Join-Path $env:LOCALAPPDATA "Programs\DIA_System"
$targetExe = Join-Path $installDir "DIA_System.exe"
$startMenuDir = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\DIA_System"
$desktopShortcut = Join-Path ([Environment]::GetFolderPath("Desktop")) "DIA System.lnk"
$startShortcut = Join-Path $startMenuDir "DIA System.lnk"
$uninstallScript = Join-Path $installDir "uninstall.ps1"
$uninstallReg = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\DIA_System"

if (!(Test-Path -LiteralPath $sourceExe)) {
    throw "Installer payload is missing DIA_System.exe."
}

New-Item -ItemType Directory -Force -Path $installDir | Out-Null
New-Item -ItemType Directory -Force -Path $startMenuDir | Out-Null

Copy-Item -LiteralPath $sourceExe -Destination $targetExe -Force

$uninstallContent = @'
$ErrorActionPreference = "Stop"
$appName = "DIA_System"
$installDir = Join-Path $env:LOCALAPPDATA "Programs\DIA_System"
$startMenuDir = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\DIA_System"
$desktopShortcut = Join-Path ([Environment]::GetFolderPath("Desktop")) "DIA System.lnk"
$uninstallReg = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\DIA_System"

Remove-Item -LiteralPath $desktopShortcut -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $startMenuDir -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $uninstallReg -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $installDir -Recurse -Force -ErrorAction SilentlyContinue
'@
Set-Content -LiteralPath $uninstallScript -Value $uninstallContent -Encoding UTF8

$shell = New-Object -ComObject WScript.Shell
foreach ($shortcutPath in @($desktopShortcut, $startShortcut)) {
    $shortcut = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = $targetExe
    $shortcut.WorkingDirectory = $installDir
    $shortcut.IconLocation = "$targetExe,0"
    $shortcut.Description = $displayName
    $shortcut.Save()
}

New-Item -Path $uninstallReg -Force | Out-Null
Set-ItemProperty -Path $uninstallReg -Name DisplayName -Value $displayName
Set-ItemProperty -Path $uninstallReg -Name DisplayVersion -Value "1.0.0"
Set-ItemProperty -Path $uninstallReg -Name Publisher -Value "DIA"
Set-ItemProperty -Path $uninstallReg -Name InstallLocation -Value $installDir
Set-ItemProperty -Path $uninstallReg -Name DisplayIcon -Value $targetExe
Set-ItemProperty -Path $uninstallReg -Name UninstallString -Value "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$uninstallScript`""
Set-ItemProperty -Path $uninstallReg -Name NoModify -Value 1 -Type DWord
Set-ItemProperty -Path $uninstallReg -Name NoRepair -Value 1 -Type DWord

Write-Host "$displayName installed successfully."
