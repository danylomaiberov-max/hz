#!/bin/sh
# Fetches the pinned llama.cpp revision into third_party/llama.cpp and applies the Windows XP patch
# (replaces SRWLOCK / CONDITION_VARIABLE / InitOnce, which need Vista or newer).
set -e
cd "$(dirname "$0")/.."
REV=23b0202a189c44a54625aadcb37a946dd1d6278d
if [ ! -d third_party/llama.cpp/.git ]; then
  git clone https://github.com/ggml-org/llama.cpp third_party/llama.cpp
fi
cd third_party/llama.cpp
git fetch --depth 1 origin "$REV" 2>/dev/null || git fetch origin
git checkout -f "$REV"
git apply --whitespace=nowarn ../llama-winxp.patch
echo "llama.cpp ready at $(git rev-parse --short HEAD) with XP patch"
