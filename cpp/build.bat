@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1

cd /d d:\github\quant\Crypto-Data-Engine

if exist cpp\bar_aggregator\build rmdir /s /q cpp\bar_aggregator\build

echo === CMake Configure ===
cmake -B cpp\bar_aggregator\build -S cpp\bar_aggregator -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release > cpp\build_log.txt 2>&1
type cpp\build_log.txt

echo === CMake Build ===
cmake --build cpp\bar_aggregator\build > cpp\build_log2.txt 2>&1
type cpp\build_log2.txt

if exist bin\bar_aggregator.exe (
    echo === Build SUCCESS ===
    dir bin\bar_aggregator.exe
) else (
    echo === Build FAILED ===
    exit /b 1
)
