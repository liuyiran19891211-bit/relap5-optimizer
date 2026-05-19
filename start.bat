@echo off
cd /d "%~dp0"

copy /Y "indta.i" "indta"
if exist "outdta" del /Q "outdta"
if exist "rstplt" del /Q "rstplt"

"relap.exe"

copy /Y "outdta" "outdta.o"
copy /Y "rstplt" "rstplt.r"
