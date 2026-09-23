# 8. Kaleidoscope: Compiling to Object Code

## Chapter 8 Introduction

Welcome to Chapter 8 of the "[Implementing a language with MLIR](chapter-00.md)" tutorial. This chapter describes how to compile our
language down to object files.

## Choosing a target

Up to this point, MLIR has provided the representations and transformations used to progressively lower our source language. To produce a native object file, we now cross the boundary into LLVM’s target infrastructure. We translate our lowered MLIR module into LLVM IR, then let LLVM select the target machine and emit code for it.

LLVM supports native and cross-target code generation. By default, our compiler
targets the current machine, but object-emission mode also accepts a target
triple through the `--target` option.

To specify the architecture that you want to target, we use a string
called a "target triple". This takes the form
`<arch><sub>-<vendor>-<sys>-<abi>` (see the [cross compilation docs](https://clang.llvm.org/docs/CrossCompilation.html#target-triple)).

As an example, we can see what clang thinks is our current target
triple:

```
$ clang --version | grep Target
Target: arm64-apple-darwin25.6.0
```

Running this command may show something different on your machine as
you might be using a different architecture or operating system to me.

Fortunately, we don't need to hard-code a target triple. LLVM provides
`sys::getDefaultTargetTriple`, which returns the triple of the current machine.
We use it when the user does not supply `--target`:

```cpp
static llvm::cl::opt<std::string>
    TargetTripleOption("target",
                       llvm::cl::desc("Target triple for object emission"),
                       llvm::cl::value_desc("triple"), llvm::cl::init(""));

std::string TargetTriple =
    TargetTripleOption.empty()
        ? llvm::sys::getDefaultTargetTriple()
        : llvm::Triple::normalize(TargetTripleOption);
```

LLVM doesn't require us to link in all the target
functionality. For example, if we're just using the JIT, we don't need
the assembly printers. Similarly, if we're only targeting certain
architectures, we can only link in the functionality for those
architectures.

Because `--target` can select a different architecture, we initialize every
LLVM target linked into the compiler:

```cpp
llvm::InitializeAllTargetInfos();
llvm::InitializeAllTargets();
llvm::InitializeAllTargetMCs();
llvm::InitializeAllAsmParsers();
llvm::InitializeAllAsmPrinters();
```

We can now use our target triple to get a `Target`:

```cpp
std::string Error;
const llvm::Target *Target =
    llvm::TargetRegistry::lookupTarget(TargetTriple, Error);

// Print an error and exit if we couldn't find the requested target.
// This generally occurs if we've forgotten to initialise the
// TargetRegistry or we have a bogus target triple.
if (!Target) {
  llvm::errs() << Error << '\n';
  return 1;
}
```

## Target Machine

We will also need a `TargetMachine`. This class provides a complete
machine description of the machine we're targeting. If we want to
target a specific feature (such as SSE) or a specific CPU (such as
Intel's Sandylake), we do so now.

To see which features and CPUs LLVM knows about for the host target built into
our LLVM installation, we can use `llc` with the triple reported by
`llvm-config`:

```bash
$ llvm-as < /dev/null \
    | llc -mtriple="$(llvm-config --host-target)" -mattr=help
Available CPUs for this target:

  a64fx         - Select the a64fx processor.
  apple-m1      - Select the apple-m1 processor.
  apple-m2      - Select the apple-m2 processor.
  ...

Available features for this target:

  aes                   - Enable AES support.
  crc                   - Enable CRC support.
  neon                  - Enable NEON instructions.
  ...
```

The exact list depends on which backends were enabled when LLVM was built. You
can check them with `llvm-config --targets-built`; this tutorial's local LLVM
build, for example, reports only `AArch64`.

For our example, we'll use the generic CPU without additional target features
and request position-independent code:

```cpp
llvm::TargetOptions Options;
std::unique_ptr<llvm::TargetMachine> TargetMachine(
    Target->createTargetMachine(llvm::Triple(TargetTriple), "generic", "",
                                Options, llvm::Reloc::PIC_));

if (!TargetMachine) {
  llvm::errs() << "Could not create the target machine\n";
  return 1;
}
```

## Configuring the Module

We're now ready to configure our module, to specify the target and
data layout. This isn't strictly necessary, but the [frontend
performance guide](https://llvm.org/docs/Frontend/PerformanceTips.html) recommends
this. Optimizations benefit from knowing about the target and data
layout.

```cpp
auto Lowered = ExitOnErr(lowerToLLVM(TargetMachine->createDataLayout()));
Lowered.Module->setTargetTriple(llvm::Triple(TargetTriple));
```

## Emit Object Code

We're ready to emit object code. The conventional `-o` option lets the user
choose the output filename. When it is omitted, we replace the input file's
extension with `.o`, so `average.ks` produces `average.o`:

```cpp
static llvm::cl::opt<std::string>
    OutputFilename("o", llvm::cl::desc("Output filename"),
                   llvm::cl::value_desc("filename"), llvm::cl::init(""));

llvm::SmallString<256> Filename;
if (OutputFilename.empty()) {
  Filename = InputFilename;
  llvm::sys::path::replace_extension(Filename, "o");
} else {
  Filename = OutputFilename;
}

std::error_code EC;
llvm::raw_fd_ostream Dest(Filename, EC, llvm::sys::fs::OF_None);

if (EC) {
  llvm::errs() << "Could not open " << Filename << ": " << EC.message()
               << '\n';
  return 1;
}
```

Finally, we define a pass that emits object code, then we run that
pass:

```cpp
llvm::legacy::PassManager EmitPM;

if (TargetMachine->addPassesToEmitFile(
        EmitPM, Dest, nullptr, llvm::CodeGenFileType::ObjectFile)) {
  llvm::errs() << "Target machine cannot emit an object file\n";
  return 1;
}

EmitPM.run(*Lowered.Module);
Dest.flush();
llvm::outs() << "Wrote " << Filename << '\n';
```

## Putting It All Together

Does it work? Let's give it a try. Here is the CMake configuration:

```cmake(../code/chapter-08/CMakeLists.txt)
```

Configure and build it in the same way as the previous chapters:

```bash
$ cmake -S . -B build \
    -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir
$ cmake --build build
```

With no arguments, `toy` remains the interactive JIT REPL. Object emission is
a separate batch mode: `--emit-object` requires a source filename, reads the
complete file, and writes an object file without displaying REPL prompts.

Save a simple `average` function in `average.ks`:

```kaleidoscope
def average(x y) (x + y) * 0.5;
```

Then compile it:

```text
$ ./build/toy --emit-object average.ks
Wrote average.o
```

We have an object file! To test it, let's write a simple program and
link it with our output. Here's the source code:

```cpp
// main.cpp
#include <iostream>

extern "C" {
    double average(double, double);
}

int main() {
    std::cout << "average of 3.0 and 4.0: " << average(3.0, 4.0) << std::endl;
}
```

We link our program to `average.o` and check the result is what we
expected:

```text
$ clang++ main.cpp average.o -o main
$ ./main
average of 3.0 and 4.0: 3.5
```

On macOS, if `clang++` cannot find the standard library headers, run the link
command through Xcode's toolchain wrapper instead:

```text
xcrun clang++ main.cpp average.o -o main
```

## Output Names and Cross-Compilation

The default command above builds `average.o` for the current machine. Use `-o`
when you want a different output path:

```text
$ ./build/toy --emit-object average.ks -o result.o
Wrote result.o
```

To request a different target, supply its triple:

```text
$ ./build/toy --emit-object average.ks \
    --target=aarch64-unknown-linux-gnu
Wrote average.o
```

You can inspect the emitted object with `llvm-readobj`:

```text
$ llvm-readobj --file-headers average.o

File: average.o
Format: elf64-littleaarch64
Arch: aarch64
AddressSize: 64bit
LoadName: <Not found>
ElfHeader {
  ...
}
```

Here the output confirms that LLVM emitted an AArch64 ELF object rather than a
native macOS object. Producing an object for another target does not by itself
provide that target's linker, system libraries, or sysroot; those are still
needed to link the object into a complete executable. The requested backend
must also be present in the LLVM build used to compile `toy`.

## Full Code Listing

```cpp(../code/chapter-08/toy.cpp)
```


[Next: Adding Debug Information](chapter-09.md)
