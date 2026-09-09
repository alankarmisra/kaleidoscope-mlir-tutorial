#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <input.mlir>" >&2
  exit 1
fi

input_file=$1

if [[ ! -f "$input_file" ]]; then
  echo "error: input file does not exist: $input_file" >&2
  exit 1
fi

# Override this when LLVM is built elsewhere:
#   KALEIDOSCOPE_LLVM_BUILD=/path/to/llvm-build ./compile.sh input.mlir
kaleidoscope_llvm_build="${KALEIDOSCOPE_LLVM_BUILD:-/Users/alankar/Documents/opensource/llvm-project/build}"
mlir_opt="$kaleidoscope_llvm_build/bin/mlir-opt"
mlir_translate="$kaleidoscope_llvm_build/bin/mlir-translate"
llc="$kaleidoscope_llvm_build/bin/llc"

clang_arguments=()
if [[ $(uname -s) == Darwin ]]; then
  clang="${CLANG:-$(xcrun --find clang)}"
  macos_sdk=$(xcrun --sdk macosx --show-sdk-path)
  clang_arguments+=("-isysroot" "$macos_sdk")
else
  clang="${CLANG:-clang}"
fi

for tool in "$mlir_opt" "$mlir_translate" "$llc" "$clang"; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "error: required tool not found: $tool" >&2
    exit 1
  fi
done

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
output_dir="$script_dir/build"
input_name=$(basename -- "$input_file")
program_name=${input_name%.mlir}

lowered_mlir="$output_dir/$program_name.lowered.mlir"
llvm_ir="$output_dir/$program_name.ll"
object_file="$output_dir/$program_name.o"
executable="$output_dir/$program_name"

mkdir -p "$output_dir"

"$mlir_opt" "$input_file" \
  --convert-arith-to-llvm \
  --convert-func-to-llvm \
  --reconcile-unrealized-casts \
  -o "$lowered_mlir"

"$mlir_translate" \
  --mlir-to-llvmir \
  "$lowered_mlir" \
  -o "$llvm_ir"

"$llc" -filetype=obj "$llvm_ir" -o "$object_file"

"$clang" "${clang_arguments[@]}" "$object_file" -o "$executable"

echo "lowered MLIR: $lowered_mlir"
echo "LLVM IR:     $llvm_ir"
echo "object file: $object_file"
echo "executable:  $executable"
