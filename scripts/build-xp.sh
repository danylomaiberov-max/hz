#!/bin/sh
# Cross-compiles ollama.exe (32-bit, Windows XP SP2+) on Linux with MinGW-w64.
#   apt install cmake g++-mingw-w64-i686-posix
# Pass -DXPLLM_SSE2=OFF for CPUs older than Pentium 4 / Athlon 64.
set -e
cd "$(dirname "$0")/.."
[ -f third_party/llama.cpp/CMakeLists.txt ] || sh scripts/fetch-llama.sh
cmake -S . -B build-xp -DCMAKE_TOOLCHAIN_FILE=cmake/mingw-i686-xp.cmake -DCMAKE_BUILD_TYPE=Release "$@"
cmake --build build-xp -j"$(nproc)" --target ollama
i686-w64-mingw32-strip build-xp/ollama.exe
ls -l build-xp/ollama.exe
