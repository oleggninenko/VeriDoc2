@echo off
REM ================================
REM Install mkcert rootCA.pem into Windows Trusted Root store
REM ================================

set CERTFILE=%~dp0rootCA.pem

if not exist "%CERTFILE%" (
    echo [ERROR] rootCA.pem not found in the same folder as this script.
    pause
    exit /b 1
)

echo Installing root CA certificate...
REM Use certutil to add it to LocalMachine\Root
certutil -addstore -f "Root" "%CERTFILE%"

if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] Root CA installed successfully.
) else (
    echo [ERROR] Failed to install root CA. Run this script as Administrator.
)

pause
