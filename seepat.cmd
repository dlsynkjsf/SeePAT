@echo off
setlocal
pushd "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo SeePAT environment missing: create .venv and install the project first.
  popd
  exit /b 1
)
set "PYTHONPATH=%CD%\src;%CD%;%PYTHONPATH%"
".venv\Scripts\python.exe" -u -m scripts.seepat_cli %*
set "result=%errorlevel%"
popd
exit /b %result%
