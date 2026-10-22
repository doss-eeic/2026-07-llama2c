if not exist build mkdir build
cl.exe /fp:fast /Ox /openmp /I. run/run.c win.c /Fe:build/run-fused.exe
