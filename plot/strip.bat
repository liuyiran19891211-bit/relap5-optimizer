copy rstplt rstplt.r
del outdta
del stripf
copy indta indta.backup
copy strip.i indta
relap.exe
copy indta.backup indta
del indta.backup
