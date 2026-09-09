# Kaleidoscope with MLIR

An incremental compiler tutorial that rebuilds LLVM's Kaleidoscope language
with MLIR.

The first part follows the concepts and progression of LLVM's *My First
Language Frontend with LLVM* tutorial while replacing direct LLVM IR generation
with MLIR dialects, passes, and progressive lowering. This gives readers who
already understand Kaleidoscope a concrete bridge from LLVM's APIs to MLIR's
architecture.

The second part extends the scalar language with tensors and connects the
result to MLIR's Toy tutorial. A custom dialect is introduced only when the
language gains semantics that upstream dialects cannot represent honestly.

## Objectives

- Preserve Kaleidoscope's recognizable syntax, examples, and feature sequence.
- Map familiar LLVM concepts to their MLIR counterparts.
- Construct and verify MLIR with the C++ API.
- Use upstream dialects such as `builtin`, `arith`, `func`, and `scf` before
  introducing a custom dialect.
- Inspect the IR between transformations instead of hiding the pass pipeline.
- Lower progressively to the LLVM dialect, translate to LLVM IR, emit an object
  file, and link a native executable.
- Cover functions, calls, optimization, control flow, mutable variables,
  user-defined operators, object generation, and debug information.
- Add tensors, shape information, rewrite patterns, and dialect conversion as a
  bridge to the MLIR Toy tutorial.
- Keep every chapter as an independent, buildable compiler snapshot with LLVM
  lit tests.

## Initial pipeline

```text
Kaleidoscope source
  -> lexer
  -> parser
  -> AST
  -> MLIR dialects
  -> LLVM dialect
  -> LLVM IR
  -> object file
  -> system linker
  -> executable
```

## Repository layout

```text
.
├── docs/
│   └── chapter-XX.md
├── code/
│   └── chapter-XX/
│       ├── kaleidoscope.cpp
│       ├── CMakeLists.txt
│       ├── build.sh
│       ├── compile.sh
│       └── test/
└── README.md
```

## Attribution

This project is inspired by and derived in structure from LLVM's
[Kaleidoscope tutorial](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/)
and connects to MLIR's [Toy tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/).
It is not an official LLVM project.

## Status

Bootstrap phase. Chapter 1 currently proves the MLIR-to-native backend while
the Kaleidoscope frontend is being introduced.
