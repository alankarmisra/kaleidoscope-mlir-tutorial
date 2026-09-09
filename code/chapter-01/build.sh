#!/usr/bin/env bash

set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
build_dir="$script_dir/build"

# Override this when LLVM is built somewhere else:
#   KALEIDOSCOPE_LLVM_BUILD=/path/to/llvm-build ./build.sh
kaleidoscope_llvm_build="${KALEIDOSCOPE_LLVM_BUILD:-/Users/alankar/Documents/opensource/llvm-project/build}"

cmake \
  -S "$script_dir" \
  -B "$build_dir" \
  -G Ninja \
  -DMLIR_DIR="$kaleidoscope_llvm_build/lib/cmake/mlir" \
  -DLLVM_DIR="$kaleidoscope_llvm_build/lib/cmake/llvm"

cmake --build "$build_dir"

source_mlir="$build_dir/source.mlir"
"$build_dir/kaleidoscope" "$script_dir/answer.ks" > "$source_mlir"

KALEIDOSCOPE_LLVM_BUILD="$kaleidoscope_llvm_build" \
  "$script_dir/compile.sh" "$source_mlir"
