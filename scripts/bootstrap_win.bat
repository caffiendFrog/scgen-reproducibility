@echo off
REM Convenience batch file to call the PowerShell bootstrap script

powershell.exe -ExecutionPolicy Bypass -File "%~dp0bootstrap_win.ps1"
