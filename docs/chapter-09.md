# 9. Kaleidoscope: Adding Debug Information

## Chapter 9 Introduction

Welcome to Chapter 9 of the "[Implementing a language with MLIR](chapter-00.md)" tutorial. In chapters 1 through 8, we've built a
decent little programming language with functions and variables.
What happens if something goes wrong though, how do you debug your
program?

Source level debugging uses formatted data that helps a debugger
translate from binary and the state of the machine back to the
source that the programmer wrote. In LLVM we generally use a format
called [DWARF](http://dwarfstd.org). DWARF is a compact encoding
that represents types, source locations, and variable locations.

The short summary of this chapter is that we'll go through the
various things you have to add to a programming language to
support debug info, and how you translate that into DWARF.

Here's the sample program we'll be compiling:

```kaleidoscope
def fib(x)
  if x < 3 then
    1
  else
    fib(x-1)+fib(x-2);

fib(10)
```

## Why is this a hard problem?

Debug information is a hard problem for a few different reasons - mostly
centered around optimized code. First, optimization makes keeping source
locations more difficult. MLIR operations carry source locations, which are
preserved through lowering and ultimately become LLVM IR debug locations.
Optimization passes should keep the source locations for newly created
instructions, but merged instructions only get to keep a single location -
this can cause jumping around when stepping through optimized programs.
Secondly, optimization can move variables in ways that are either optimized
out, shared in memory with other variables, or difficult to track.

Kaleidoscope enables optimization by default. When you want predictable
source-level stepping, compile with `-O0`:

```text
$ ./build/toy --emit-object -O0 fib.ks
Wrote fib.o
```

Use `-O1`, `-O2`, or `-O3` to enable the canonicalization and CSE passes from
Chapter 4 together with the corresponding LLVM code-generation optimization
level. If you omit the option, Kaleidoscope uses `-O2`.

We accept the optimization level using LLVM's command-line library and default
to `-O2`:

```cpp
static llvm::cl::opt<char>
    OptLevel("O", llvm::cl::desc("Optimization level: -O0, -O1, -O2, or -O3"),
             llvm::cl::Prefix, llvm::cl::init('2'));
```

When initializing the MLIR pass manager, we add the optimization passes at
every level above `-O0`:

```cpp
// Create a pass manager and enable our simple optimizations above -O0.
ThePM = std::make_unique<PassManager>(TheContext.get());
if (OptLevel != '0') {
  ThePM->addNestedPass<func::FuncOp>(createCanonicalizerPass());
  ThePM->addNestedPass<func::FuncOp>(createCSEPass());
}
```

Finally, we pass the corresponding code-generation optimization level to
LLVM's target machine:

```cpp
llvm::TargetOptions Options;
llvm::CodeGenOptLevel CodeGenOpt;
switch (OptLevel) {
case '0': CodeGenOpt = llvm::CodeGenOptLevel::None; break;
case '1': CodeGenOpt = llvm::CodeGenOptLevel::Less; break;
case '2': CodeGenOpt = llvm::CodeGenOptLevel::Default; break;
case '3': CodeGenOpt = llvm::CodeGenOptLevel::Aggressive; break;
}
std::unique_ptr<llvm::TargetMachine> TargetMachine(
    Target->createTargetMachine(llvm::Triple(TargetTriple), "generic", "",
                                Options, llvm::Reloc::PIC_, std::nullopt,
                                CodeGenOpt));
```

## Compile Unit

The top level container for a section of code in DWARF is a compile unit.
This contains the type and function data for an individual translation unit
(read: one file of source code). The debug information for the functions in
`fib.ks`, for example, belongs to a single compile unit.

## Source Locations

The most important thing for debug information is accurate source location -
this makes it possible to map your source code back. We have a problem though,
Kaleidoscope really doesn't have any source location information in the lexer
or parser so we'll need to add it.

```cpp
struct SourceLocation {
  int Line;
  int Col;
};
static SourceLocation CurLoc;
static SourceLocation LexLoc = {1, 0};

static int advance() {
  int LastChar = getchar();
  if (LastChar == '\n' || LastChar == '\r') {
    ++LexLoc.Line;
    LexLoc.Col = 0;
  } else {
    ++LexLoc.Col;
  }
  return LastChar;
}
```

In this set of code we've added some functionality on how to keep track of the
line and column of the "source file". As we lex every token we set our
current "lexical location" to the assorted line and column for the beginning
of the token. We do this by overriding all of the previous calls to
`getchar()` with our new `advance()` that keeps track of the information
and then we have added to all of our AST classes a source location:

```cpp
class ExprAST {
  SourceLocation Loc;

public:
  ExprAST(SourceLocation Loc = CurLoc) : Loc(Loc) {}
  virtual ~ExprAST() = default;

  virtual Value codegen() = 0;
  virtual const std::string *getVariableName() const { return nullptr; }
  SourceLocation getSourceLocation() const { return Loc; }
};
```

We turn the current AST location into a `FileLineColLoc` with this helper:

```cpp
static SourceLocation CodegenLoc = {1, 1};

static Location getLocation() {
  llvm::StringRef Filename = "<stdin>";
  if (!InputFilename.empty())
    Filename = InputFilename.getValue();
  return FileLineColLoc::get(TheContext.get(), Filename, CodegenLoc.Line,
                             CodegenLoc.Col);
}
```

A small guard selects an AST node's location while its code is generated and
restores the previous location afterward:

```cpp
class LocationGuard {
  SourceLocation Previous;

public:
  explicit LocationGuard(SourceLocation Loc) : Previous(CodegenLoc) {
    CodegenLoc = Loc;
  }
  ~LocationGuard() { CodegenLoc = Previous; }
};
```

Each code-generation method creates one of these guards before it creates any
MLIR operations. For example:

```cpp
Value NumberExprAST::codegen() {
  LocationGuard Guard(getSourceLocation());
  return TheBuilder->create<arith::ConstantOp>(
      getLocation(), TheBuilder->getF64FloatAttr(Val));
}
```

The location is now part of the MLIR operation. Conversion passes carry it
through the lowering pipeline, and the final LLVM translation turns it into a
`DILocation` attached to the generated instruction.

## DWARF Emission Setup

After lowering our higher-level operations to the LLVM dialect, we describe
the source file with a compile unit. We identify the language as C, which suits
Kaleidoscope because we follow the C calling convention and ABI. The
optimization flag comes directly from the option selected by the user:

```cpp
auto File = InputPath.empty()
                ? LLVM::DIFileAttr::get(Context, "<stdin>", "")
                : LLVM::DIFileAttr::get(
                      Context, llvm::sys::path::filename(InputPath),
                      llvm::sys::path::parent_path(InputPath));
auto CompileUnit = LLVM::DICompileUnitAttr::get(
    DistinctAttr::create(UnitAttr::get(Context)), llvm::dwarf::DW_LANG_C,
    File, StringAttr::get(Context, "Kaleidoscope"),
    /*isOptimized=*/OptLevel != '0', LLVM::DIEmissionKind::Full);
Module->setLoc(FusedLoc::get(Context, {Module.getLoc()}, CompileUnit));
```

We also attach a subprogram description to every lowered function. Definitions
are marked with `Definition`, while optimized builds additionally use the
`Optimized` flag:

```cpp
for (LLVM::LLVMFuncOp Function : Module.getOps<LLVM::LLVMFuncOp>()) {
  Location OriginalLoc = Function.getLoc();
  LLVM::DIFileAttr FunctionFile = File;
  int64_t Line = 1;
  if (auto FileLoc = OriginalLoc->findInstanceOf<FileLineColLoc>()) {
    llvm::StringRef FunctionPath = FileLoc.getFilename().getValue();
    FunctionFile = LLVM::DIFileAttr::get(
        Context, llvm::sys::path::filename(FunctionPath),
        llvm::sys::path::parent_path(FunctionPath));
    Line = FileLoc.getLine();
  }

  DistinctAttr Id;
  LLVM::DICompileUnitAttr FunctionCompileUnit = CompileUnit;
  auto Flags = static_cast<LLVM::DISubprogramFlags>(0);
  if (OptLevel != '0')
    Flags = Flags | LLVM::DISubprogramFlags::Optimized;
  if (Function.isExternal()) {
    FunctionCompileUnit = {};
  } else {
    Id = DistinctAttr::create(UnitAttr::get(Context));
    Flags = Flags | LLVM::DISubprogramFlags::Definition;
  }

  auto FunctionType = LLVM::DISubroutineTypeAttr::get(
      Context, llvm::dwarf::DW_CC_normal, {});
  auto Name = Function.getNameAttr();
  auto Scope = LLVM::DISubprogramAttr::get(
      Context, Id, FunctionCompileUnit, FunctionFile, Name, Name,
      FunctionFile, Line, Line, Flags, FunctionType,
      /*retainedNodes=*/{}, /*annotations=*/{});
  Function->setLoc(FusedLoc::get(Context, {OriginalLoc}, Scope));
}
```

We then run MLIR's debug-scope pass. Because the compile unit and subprograms
already exist, the pass preserves them and adds the remaining scope information
to our operation locations. The information is later translated into LLVM IR
debug metadata automatically.

```cpp
LLVM::DIScopeForLLVMFuncOpPassOptions DebugOptions;
DebugOptions.emissionKind = LLVM::DIEmissionKind::Full;
PassManager DebugPM(TheContext.get());
DebugPM.addPass(
    LLVM::createDIScopeForLLVMFuncOpPass(std::move(DebugOptions)));
if (failed(DebugPM.run(*TheModule)))
  return llvm::make_error<llvm::StringError>(
      "could not add LLVM debug scopes",
      llvm::inconvertibleErrorCode());
```

## Variables

Now that we have functions, we need to be able to print out the variables we have in scope. Let's get our function arguments set up so we can get decent backtraces and see how our functions are being called. MLIR represents values in SSA form and does not preserve our source-level variable names. A source location tells the debugger where an operation came from, but not that a value represents a variable named `x`. To make function arguments visible by name in the debugger, we need to attach that information explicitly.

```cpp
FunctionParameters[P.getName()] = P.getArgs();
```

After lowering to the LLVM dialect, we can attach debug information to each variable declaration. This records the variable's name, type, source location, and storage location so it remains visible to the debugger.

```cpp
MLIRContext *Context = Module.getContext();
auto DoubleType = LLVM::DIBasicTypeAttr::get(
    Context, llvm::dwarf::DW_TAG_base_type, "double", 64,
    llvm::dwarf::DW_ATE_float);
auto EmptyExpression = LLVM::DIExpressionAttr::get(Context);
```

Memref lowering represents the mutable storage with an LLVM pointer wrapped in
a descriptor. With optimization, that descriptor is usually simplified away.
At `-O0`, however, the address used by the store may still be produced by a
chain of `llvm.insertvalue` and `llvm.extractvalue` operations. A debug
declaration attached to that derived value does not produce a stable DWARF
location for the parameter.

We therefore follow the operands of the value until we reach the underlying
`llvm.alloca`:

```cpp
static Value findUnderlyingAlloca(Value V) {
  Operation *DefiningOp = V.getDefiningOp();
  if (!DefiningOp)
    return {};
  if (isa<LLVM::AllocaOp>(DefiningOp))
    return V;
  for (Value Operand : DefiningOp->getOperands())
    if (Value Alloca = findUnderlyingAlloca(Operand))
      return Alloca;
  return {};
}
```

For each parameter, we use its function scope and lowered stack address to
create the variable description and declaration:

```cpp
auto Variable = LLVM::DILocalVariableAttr::get(
    Scope, Name, Scope.getFile(), Scope.getLine(), ArgumentNumber + 1,
    /*alignInBits=*/0, DoubleType, LLVM::DIFlags::Zero);
Value VariableAddress = findUnderlyingAlloca(ArgumentStore.getAddr());
if (!VariableAddress)
  VariableAddress = ArgumentStore.getAddr();
OpBuilder Builder(ArgumentStore);
Builder.setInsertionPointAfter(ArgumentStore);
Builder.create<LLVM::DbgDeclareOp>(ArgumentStore.getLoc(),
                                   VariableAddress, Variable,
                                   EmptyExpression);
```

`DILocalVariableAttr` records the scope, name, source location, argument
number, and type. `DbgDeclareOp` says that the variable lives at the recovered
stack address. During translation this becomes LLVM's `#dbg_declare` record.
Without the recovery, the declaration refers to the descriptor extraction:

```llvm
#dbg_declare(ptr %descriptor_extract, ...)
```

With the underlying allocation, it instead refers to the actual stack slot:

```llvm
#dbg_declare(ptr %alloca, ...)
```

This is what allows the parameter to remain visible in the emitted DWARF when
we compile with `-O0`.

With this we have enough debug information to set breakpoints in functions,
print their arguments, and inspect the call stack. The source locations and
variable descriptions were both created in MLIR; translating the LLVM dialect
produces the final LLVM debug metadata.

## Inspecting the Debug Information

We can now compile `fib.ks` to an object file:

```text
$ ./build/toy --emit-object -O0 fib.ks
Wrote fib.o
```

The `llvm-dwarfdump` tool lets us inspect the DWARF information in that object.
On my Apple silicon Mac, the relevant part of the output looks like this:

```text
$ llvm-dwarfdump --debug-info fib.o
fib.o: file format Mach-O arm64
...
0x00000026:   DW_TAG_subprogram
                DW_AT_low_pc              (0x0000000000000000)
                DW_AT_high_pc             (0x0000000000000090)
                DW_AT_APPLE_omit_frame_ptr (true)
                DW_AT_frame_base          (DW_OP_reg31 WSP)
                DW_AT_linkage_name        ("fib")
                DW_AT_name                ("fib")
                DW_AT_decl_file           ("fib.ks")
                DW_AT_decl_line           (1)
                DW_AT_external            (true)

0x0000003f:     DW_TAG_formal_parameter
                  DW_AT_location  (DW_OP_fbreg +24)
                  DW_AT_name      ("x")
                  DW_AT_decl_file ("fib.ks")
                  DW_AT_decl_line (1)
                  DW_AT_type      (0x0000004e "double")

0x0000004e:   DW_TAG_base_type
                DW_AT_name      ("double")
                DW_AT_encoding  (DW_ATE_float)
                DW_AT_byte_size (0x08)
...
```

Offsets, addresses, and some target-specific attributes will differ between
systems. The important parts are the `fib` subprogram, its source file and
line, and the parameter named `x` with type `double`.

## Full Code Listing

Here is the complete code listing for our running example, enhanced with
debug information. Here is the CMake configuration:

```cmake(../code/chapter-09/CMakeLists.txt)
```

To build this example, use:

```bash
cmake -S . -B build \
  -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir
cmake --build build
./build/toy
```

Here is the code:

```cpp(../code/chapter-09/toy.cpp)
```

[Next: Conclusion and other useful LLVM tidbits](chapter-10.md)
