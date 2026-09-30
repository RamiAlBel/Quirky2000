@echo off
rem Rebuild the engine (MSVC 2022 Build Tools). "build.bat perft" builds the move-generator test instead.
rem "build.bat lnn2 1024" builds for LNN2 nets of width 1024 (default: LNN1, 256, the original lite.nnue).
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d "%~dp0"
if not exist obj mkdir obj
set FLAGS=/nologo /O2 /Oi /Ot /GL /arch:AVX2 /std:c++17 /EHsc /DNDEBUG /W3 /wd4996 /DUSE_PEXT /Foobj\
if "%1"=="perft" (
  cl %FLAGS% perft.cpp position.cpp /Fe:perft.exe /link /LTCG || exit /b 1
  exit /b 0
)
if "%1"=="lnn2" set FLAGS=%FLAGS% /DNNUE_LNN2 /DACC_WIDTH=%2
cl %FLAGS% main.cpp search.cpp nnue.cpp position.cpp /Fe:..\nnue_engine.exe /link /LTCG || exit /b 1
echo Built ..\nnue_engine.exe
