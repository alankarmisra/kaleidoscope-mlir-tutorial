# 10. Kaleidoscope: Building a Custom Dialect

## Chapter 10 Introduction

Welcome to Chapter 10 of the "[Implementing a language with MLIR](chapter-00.md)" tutorial. So far, our AST has generated operations from MLIR's existing dialects directly. This has served us well: `arith` represents arithmetic, `func` represents functions, `scf` represents structured control flow, and `memref` provides storage for mutable variables.

There are times, however, when those operations no longer describe everything the source language knows. In Chapter 9, lowering a variable to generic storage discarded its source name. We recovered that information later by walking the lowered IR. It worked, but it would be better not to lose the information in the first place.

In this chapter, we'll add a deliberately small Kaleidoscope dialect containing only the operations needed for mutable variables. The operations preserve a variable's name and source location until our own lowering pass has enough information to create both its storage and its debug declaration.

We are not moving the whole language into a custom dialect. The [MLIR Toy tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/) demonstrates that larger architecture. Here we only need enough dialect to solve the problem in front of us.

## Why Preserve Variables?

Consider this function:

```kaleidoscope
def test(x)
  var y = x in
    (y = y + 1) * y;
```

Previously, AST generation immediately turned `x` and `y` into anonymous `memref` storage. MLIR retained their source locations, but it no longer knew that the allocations represented variables named `x` and `y`. Chapter 9 saved the parameter names separately and reconstructed the connection after lowering.

Our initial MLIR will now preserve that intent directly:

```mlir
func.func @test(%arg0: f64) -> f64 {
  %cst = arith.constant 1.000000e+00 : f64
  %0 = kaleidoscope.var "x" = %arg0 {argumentNumber = 1 : i64} : f64
  %1 = kaleidoscope.read %0 : f64
  %2 = kaleidoscope.var "y" = %1 {argumentNumber = 0 : i64} : f64
  %3 = kaleidoscope.read %2 : f64
  %4 = arith.addf %3, %cst : f64
  kaleidoscope.assign %4 to %2 : f64
  %5 = kaleidoscope.read %2 : f64
  %6 = arith.mulf %4, %5 : f64
  return %6 : f64
}
```

The standard dialects still handle everything they already describe well. Only source variables use the Kaleidoscope dialect.

## Defining the Dialect

MLIR operations and types are commonly defined using the [Operation Definition Specification](https://mlir.llvm.org/docs/DefiningDialects/Operations/), or ODS. We begin by declaring our dialect and a base class for its operations:

```tablegen
def Kaleidoscope_Dialect : Dialect {
  let name = "kaleidoscope";
  let cppNamespace = "::mlir::kaleidoscope";
  let summary = "Operations that preserve Kaleidoscope variable semantics";
  let useDefaultTypePrinterParser = 1;
}

class Kaleidoscope_Op<string mnemonic, list<Trait> traits = []>
    : Op<Kaleidoscope_Dialect, mnemonic, traits>;
```

The dialect name provides the `kaleidoscope.` prefix in textual MLIR. We also define a small handle type representing a mutable source variable:

```tablegen
def Kaleidoscope_VariableType
    : TypeDef<Kaleidoscope_Dialect, "Variable"> {
  let mnemonic = "var";
  let summary = "a mutable Kaleidoscope variable";
  let assemblyFormat = "";
}
```

It is printed as `!kaleidoscope.var`. The type deliberately says only that the value is a variable. Its eventual stack representation is a lowering decision.

## Variable Operations

We need three operations. `kaleidoscope.var` declares and initializes a variable while preserving its source name. `argumentNumber` is zero for an ordinary local and one-based for a function parameter, matching DWARF's representation:

```tablegen
def Kaleidoscope_DeclareOp : Kaleidoscope_Op<"var", []> {
  let summary = "declare and initialize a mutable source variable";
  let arguments = (ins F64:$initialValue, StrAttr:$name,
                       I64Attr:$argumentNumber);
  let results = (outs Kaleidoscope_VariableType:$variable);
  let assemblyFormat = "$name `=` $initialValue attr-dict `:` type($initialValue)";
}
```

The other two operations read and update the variable:

```tablegen
def Kaleidoscope_ReadOp : Kaleidoscope_Op<"read", []> {
  let summary = "read a mutable source variable";
  let arguments = (ins Kaleidoscope_VariableType:$variable);
  let results = (outs F64:$value);
  let assemblyFormat = "$variable attr-dict `:` type($value)";
}

def Kaleidoscope_AssignOp : Kaleidoscope_Op<"assign", []> {
  let summary = "assign a new value to a mutable source variable";
  let arguments = (ins Kaleidoscope_VariableType:$variable, F64:$value);
  let assemblyFormat = "$value `to` $variable attr-dict `:` type($value)";
}
```

Notice that `kaleidoscope.read` is not marked `Pure`. Two reads of the same variable are not necessarily equal because an assignment may occur between them. Marking the operation pure would allow CSE to incorrectly reuse the earlier value.

TableGen generates the type and operation classes. The dialect registers them when it is initialized:

```cpp
void KaleidoscopeDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "KaleidoscopeTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "KaleidoscopeOps.cpp.inc"
      >();
}
```

The generated files are added to the build with `mlir_tablegen`; the complete commands appear in this chapter's `CMakeLists.txt`.

## Generating Variable IR

Creating a source variable now creates one operation containing everything the lowering will need:

```cpp
static Value CreateVariable(StringRef Name, Value InitialValue,
                            int64_t ArgumentNumber = 0) {
  return TheBuilder->create<kaleidoscope::DeclareOp>(
      getLocation(), kaleidoscope::VariableType::get(TheContext.get()),
      InitialValue, TheBuilder->getStringAttr(Name),
      TheBuilder->getI64IntegerAttr(ArgumentNumber));
}
```

A variable expression becomes a read:

```cpp
return TheBuilder->create<kaleidoscope::ReadOp>(
    getLocation(), TheBuilder->getF64Type(), It->second);
```

Assignment becomes an update while continuing to return the assigned value:

```cpp
TheBuilder->create<kaleidoscope::AssignOp>(getLocation(), It->second,
                                           AssignedValue);
return AssignedValue;
```

Function arguments use the same declaration operation, but include their one-based argument number:

```cpp
unsigned Index = 0;
for (BlockArgument Argument : TheFunction.getArguments()) {
  StringRef Name = P.getArgs()[Index];
  Value Storage = CreateVariable(Name, Argument, Index + 1);
  NamedValues[Name.str()] = Storage;
  ++Index;
}
```

Local variables and loop variables use argument number zero. We no longer need the separate `FunctionParameters` map from Chapter 9: each declaration carries its own name, location, and argument number.

## Lowering Variables and Debug Information Together

Our lowering pass converts `!kaleidoscope.var` to an LLVM pointer. A declaration becomes an `llvm.alloca` followed by the initializing `llvm.store`:

```cpp
Value One = LLVM::ConstantOp::create(
    Rewriter, Loc, Rewriter.getI64Type(), Rewriter.getI64IntegerAttr(1));
Value Address = LLVM::AllocaOp::create(Rewriter, Loc, PointerType,
                                       DoubleType, One, 0);
LLVM::StoreOp::create(Rewriter, Loc, Adaptor.getInitialValue(), Address);
```

At this exact point we still have the source variable operation and have just created its final stack address. There is nothing to rediscover. The same lowering creates its debug description and attaches it to that address:

```cpp
auto Variable = LLVM::DILocalVariableAttr::get(
    Scope, Op.getName(), Scope.getFile(), Line,
    Op.getArgumentNumber(), /*alignInBits=*/0, VariableType,
    LLVM::DIFlags::Zero);
LLVM::DbgDeclareOp::create(
    Rewriter, Loc, Address, Variable,
    LLVM::DIExpressionAttr::get(Rewriter.getContext()));
```

Reads and assignments lower directly to LLVM loads and stores:

```cpp
Rewriter.replaceOpWithNewOp<LLVM::LoadOp>(
    Op, Rewriter.getF64Type(), Adaptor.getVariable());

Rewriter.replaceOpWithNewOp<LLVM::StoreOp>(
    Op, Adaptor.getValue(), Adaptor.getVariable());
```

The conversion target declares our dialect illegal, ensuring that lowering cannot silently finish while a Kaleidoscope variable operation remains:

```cpp
ConversionTarget Target(Context);
Target.addIllegalDialect<kaleidoscope::KaleidoscopeDialect>();
Target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

RewritePatternSet Patterns(&Context);
Patterns.add<DeclareOpLowering, ReadOpLowering, AssignOpLowering>(
    Converter, &Context);
if (failed(applyPartialConversion(getOperation(), Target,
                                  std::move(Patterns))))
  signalPassFailure();
```

## Connecting the Pass Pipeline

We first lower the standard high-level dialects to the LLVM dialect. Our Chapter 9 debug pass then creates the compile unit and function scopes:

```cpp
PassManager DebugPM(TheContext.get());
DebugPM.addPass(createKaleidoscopeDebugInfoPass(
    InputFilename.getValue(), OptLevel));
```

With the function scopes available, our variable pass can create the stack storage and variable debug declarations together. MLIR's existing pass then fills in the remaining scopes on the lowered operations:

```cpp
DebugPM.addPass(std::make_unique<LowerKaleidoscopeVariablesPass>());

LLVM::DIScopeForLLVMFuncOpPassOptions DebugOptions;
DebugOptions.emissionKind = LLVM::DIEmissionKind::Full;
DebugPM.addPass(
    LLVM::createDIScopeForLLVMFuncOpPass(std::move(DebugOptions)));
```

The resulting LLVM IR contains ordinary storage plus the debug declaration:

```llvm
%x = alloca double, i64 1, align 8
store double %arg0, ptr %x, align 8
#dbg_declare(ptr %x, !variable, !DIExpression(), !location)
```

After this point the normal LLVM translation, JIT, object emitter, and DWARF generation continue unchanged.

## Trying It

The source example still executes normally:

```text
ready> def test(x) var y = x in (y = y + 1) * y;
Read function definition:
ready> test(4);
Evaluated to 25.000000
```

Using `--dump-mlir` shows the named variable operations before lowering:

<!-- code-merge:start -->
```text
$ ./build/toy --dump-mlir
ready> def test(x) var y = x in (y = y + 1) * y;
Read function definition:
```
```mlir
func.func @test(%arg0: f64) -> f64 {
  %cst = arith.constant 1.000000e+00 : f64
  %0 = kaleidoscope.var "x" = %arg0 {argumentNumber = 1 : i64} : f64
  %1 = kaleidoscope.read %0 : f64
  %2 = kaleidoscope.var "y" = %1 {argumentNumber = 0 : i64} : f64
  %3 = kaleidoscope.read %2 : f64
  %4 = arith.addf %3, %cst : f64
  kaleidoscope.assign %4 to %2 : f64
  %5 = kaleidoscope.read %2 : f64
  %6 = arith.mulf %4, %5 : f64
  return %6 : f64
}
```
<!-- code-merge:end -->

With `--dump-llvm-ir`, the variable operations have become stack allocations, loads, and stores. Each allocation also has the debug declaration generated by our lowering pass:

<!-- code-merge:start -->
```text
$ ./build/toy --dump-llvm-ir
ready> def test(x) var y = x in (y = y + 1) * y;
```
```llvm
define double @test(double %0) !dbg !3 {
  %2 = alloca double, i64 1, align 8, !dbg !6
  store double %0, ptr %2, align 8, !dbg !6
    #dbg_declare(ptr %2, !7, !DIExpression(), !6)
  %3 = load double, ptr %2, align 8, !dbg !9
  %4 = alloca double, i64 1, align 8, !dbg !10
  store double %3, ptr %4, align 8, !dbg !10
    #dbg_declare(ptr %4, !11, !DIExpression(), !10)
  %5 = load double, ptr %4, align 8, !dbg !12
  %6 = fadd double %5, 1.000000e+00, !dbg !13
  store double %6, ptr %4, align 8, !dbg !14
  %7 = load double, ptr %4, align 8, !dbg !15
  %8 = fmul double %6, %7, !dbg !16
  ret double %8, !dbg !6
}

!7 = !DILocalVariable(name: "x", arg: 1, scope: !3, file: !1, line: 1, type: !8)
!11 = !DILocalVariable(name: "y", scope: !3, file: !1, line: 1, type: !8)
```
<!-- code-merge:end -->

The first declaration describes parameter `x`; the second describes local variable `y`. The details omitted between the function and these metadata records are the compile-unit, function, type, and source-location metadata introduced in Chapter 9.

## Testing the Dialect

Dialect conversion is a particularly good place for a small regression test. This chapter includes a lit test that checks all three variable operations and then executes the result:

```bash
cmake --build build --target check-chapter-10
```

## Full Code Listing

The complete implementation is split across:

- `toy.cpp`, containing the compiler and variable-lowering pass;
- `KaleidoscopeOps.td`, containing the variable type and operation definitions;
- `KaleidoscopeDialect.h` and `KaleidoscopeDialect.cpp`, connecting the generated classes to the compiler;
- `KaleidoscopeDebugInfo.h` and `KaleidoscopeDebugInfo.cpp`, creating the compile-unit and function scopes; and
- `CMakeLists.txt`, running TableGen and building the executable.

### Build Configuration

```cmake(../code/chapter-10/CMakeLists.txt)
```

Use the chapter's build script as before:

```bash
./build.sh
```

### Compiler

```cpp(../code/chapter-10/toy.cpp)
```

### Dialect Definitions

```tablegen(../code/chapter-10/KaleidoscopeOps.td)
```

### Dialect Declaration

```cpp(../code/chapter-10/KaleidoscopeDialect.h)
```

### Dialect Implementation

```cpp(../code/chapter-10/KaleidoscopeDialect.cpp)
```

### Debug-Scope Pass Declaration

```cpp(../code/chapter-10/KaleidoscopeDebugInfo.h)
```

### Debug-Scope Pass Implementation

```cpp(../code/chapter-10/KaleidoscopeDebugInfo.cpp)
```

## Closing Thoughts

Our custom dialect is deliberately small. We introduced it because a source variable knows more than an anonymous allocation: it has a name, a source location, and perhaps an argument number. Preserving that information until lowering lets us generate storage and debug information together instead of reconstructing their relationship afterward.

This is the central reason to create a dialect. It lets a compiler retain the concepts that matter to its source language until it is ready to express them in more general operations. A future chapter could take the next step and represent the complete Kaleidoscope AST as a dialect; the MLIR Toy tutorial shows what that larger design looks like.

[Next: Conclusion and other useful LLVM tidbits](chapter-11.md)
