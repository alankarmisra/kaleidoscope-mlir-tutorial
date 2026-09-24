# Chapter 10: Building Your Own Dialect

Welcome to Chapter 10 of the "[Implementing a language with MLIR](chapter-00.md)" tutorial. So far, our AST has generated operations from MLIR's existing dialects directly. This has served us well: `arith` represents arithmetic, `func` represents functions, `scf` represents structured control flow, and `memref` provides storage for mutable variables. In this chapter, you'll create a small custom dialect called **Kaleidoscope**. We'll add a handful of operations, and they all solve one specific problem.

## The Problem We're Solving

Let's start with a simple function:

```kaleidoscope
def test(x)
  var y = x in
    (y = y + 1) * y;
```

Up to now, when our compiler saw `x` and `y`, it immediately turned them into anonymous memory allocations. They still worked as variables, but MLIR no longer knew their *names*. If you looked at the generated IR, you'd see `%0`, `%1`, `%2` — not `x` and `y`.

In Chapter 9, we worked around this by saving the parameter names separately and stitching them back together later. That worked, but it was a patch. Ideally, we wouldn't lose the information in the first place.

That's what a custom dialect gives us: a way to keep the *idea of a named variable* alive in the IR until we're ready to lower it.

Here's what the IR looks like with our new dialect:

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

Notice that everything else still uses the standard dialects you already know: `arith` for math, `func` for functions. Only the variables use `kaleidoscope.*` operations — and now the names `"x"` and `"y"` are right there in the IR.

> **A note before we go further:** We're deliberately keeping this dialect small. The [MLIR Toy tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/) shows how to move an entire language into a dialect. We just need enough to solve our specific problem.

## Step 1: Define the Dialect

MLIR dialects are usually defined with **TableGen** — a compact way to describe operations that MLIR then expands into the C++ boilerplate for you. If you've never used it, think of it as a schema language: you describe *what* you want, and TableGen generates *how* it's implemented.

Create a file called `KaleidoscopeOps.td` and start with the dialect itself:

```tablegen
include "mlir/IR/OpBase.td"

def Kaleidoscope_Dialect : Dialect {
  let name = "kaleidoscope";
  let cppNamespace = "::mlir::kaleidoscope";
  let summary = "Operations that preserve Kaleidoscope variable semantics";
  let useDefaultTypePrinterParser = 1;
}
```

Let's unpack the important fields:

- **`name`** — the prefix for every operation in this dialect. Since we set it to `"kaleidoscope"`, an operation named `var` will print as `kaleidoscope.var`. (This is the same pattern as `scf.if` or `func.call`.)
- **`cppNamespace`** — where the generated C++ classes live. Ours go in `mlir::kaleidoscope`.
- **`useDefaultTypePrinterParser`** — tells MLIR to generate parsing and printing for our types. We'll see what that looks like below.

## Step 2: Add a Variable Type

We need a way to say "this SSA value is a named variable" — distinct from "this SSA value is an `f64`." That's a **type**. Add this to your `.td` file:

```tablegen
def Kaleidoscope_VariableType
    : TypeDef<Kaleidoscope_Dialect, "Variable"> {
  let mnemonic = "var";
  let summary = "a mutable Kaleidoscope variable";
  let assemblyFormat = "";
}
```

This generates a C++ class called `mlir::kaleidoscope::VariableType` with the textual spelling `!kaleidoscope.var`. The empty `assemblyFormat` means there's nothing to print after the mnemonic — the type only says "this is a variable," nothing more. Details like how it's stored on the stack are decided later, during lowering.

> **Why keep the type so minimal?** Because it doesn't need to know. The whole point of a dialect is to say *what* something means, not *how* it will be implemented. The lowering pass will figure that out.

## Step 3: Define Three Operations

We only need three operations to handle all the variable machinery in Kaleidoscope:

1. **`kaleidoscope.var`** — declares a variable with a name
2. **`kaleidoscope.read`** — reads a variable's current value
3. **`kaleidoscope.assign`** — writes a new value into a variable

First, add a shared base class so all three operations can be defined the same way:

```tablegen
class Kaleidoscope_Op<string mnemonic, list<Trait> traits = []>
    : Op<Kaleidoscope_Dialect, mnemonic, traits>;
```

### 3a. The `var` Operation

This is the important one — it's the whole reason we're building a dialect:

```tablegen
def Kaleidoscope_DeclareOp : Kaleidoscope_Op<"var", []> {
  let summary = "declare and initialize a mutable source variable";
  let arguments = (ins F64:$initialValue, StrAttr:$name,
                       I64Attr:$argumentNumber);
  let results = (outs Kaleidoscope_VariableType:$variable);
  let assemblyFormat = "$name `=` $initialValue attr-dict `:` type($initialValue)";
}
```

Here's what each field does:

- **`"var"`** — the mnemonic. Combined with the dialect name, this produces `kaleidoscope.var`.
- **`arguments`** — everything the operation takes in:
  - `F64:$initialValue` — an SSA operand that must be an `f64`
  - `StrAttr:$name` — a string attribute holding the source name
  - `I64Attr:$argumentNumber` — a number telling DWARF which function parameter this is (`0` = local variable, `1`+ = parameter, since DWARF numbers parameters starting at 1)
- **`results`** — an SSA result of type `!kaleidoscope.var`
- **`assemblyFormat`** — the textual syntax. Backticks are literal punctuation; `$name` and `$initialValue` refer to fields above; `attr-dict` prints leftover attributes; `type($initialValue)` prints the operand's type.

The `$` names also generate C++ accessors, so you'll write `getInitialValue()`, `getName()`, and `getArgumentNumber()` in your code.

Here's the actual IR this produces for parameter `x`:

```mlir
%0 = kaleidoscope.var "x" = %arg0 {argumentNumber = 1 : i64} : f64
```

Let's read that piece by piece:

- **`%0`** — an SSA value representing the *variable itself*, not the value inside it. Subsequent `read` and `assign` operations use `%0` to refer to this variable.
- **`"x"`** — the name from the source.
- **`%arg0`** — the initial value stored in the variable (in this case, the function's first argument).
- **`{argumentNumber = 1 : i64}`** — `x` is the first parameter (DWARF uses 1-based numbering). Locals use `0`.
- **`: f64`** — the type of the *initial value*, not of `%0`. The `%0` value has type `!kaleidoscope.var`.

### 3b. The `read` and `assign` Operations

Add these next:

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

These are straightforward: `read` takes a variable and returns its `f64` value; `assign` takes a variable and an `f64` and stores the new value.

> **Important detail:** Notice that `read` is *not* marked `Pure`. Two reads of the same variable might return different values if an assignment happened in between. If we marked it pure, MLIR's common-subexpression elimination would wrongly merge the two reads into one.

## Step 4: Generate the C++ Classes

TableGen files describe operations, but your compiler needs C++ classes. The `mlir-tblgen` tool reads your `.td` file and generates header and implementation fragments. Add this to your `CMakeLists.txt`:

```cmake
set(LLVM_TARGET_DEFINITIONS KaleidoscopeOps.td)
mlir_tablegen(KaleidoscopeDialect.h.inc -gen-dialect-decls
  -dialect=kaleidoscope)
mlir_tablegen(KaleidoscopeDialect.cpp.inc -gen-dialect-defs
  -dialect=kaleidoscope)
mlir_tablegen(KaleidoscopeTypes.h.inc -gen-typedef-decls
  -typedefs-dialect=kaleidoscope)
mlir_tablegen(KaleidoscopeTypes.cpp.inc -gen-typedef-defs
  -typedefs-dialect=kaleidoscope)
mlir_tablegen(KaleidoscopeOps.h.inc -gen-op-decls)
mlir_tablegen(KaleidoscopeOps.cpp.inc -gen-op-defs)
```

Six invocations, each asking for a different piece: dialect declarations, dialect definitions, type declarations, type definitions, operation declarations, and operation definitions.

These `.inc` files are generated into your build directory (not your source tree), which is why you need:

```cmake
target_include_directories(toy PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
add_dependencies(toy KaleidoscopeOpsIncGen)
```

Now include the generated pieces in your header. Some fragments use selector macros to pick which section to include at each point:

```cpp
#include "KaleidoscopeDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "KaleidoscopeTypes.h.inc"

#define GET_OP_CLASSES
#include "KaleidoscopeOps.h.inc"
```

And in your implementation file:

```cpp
#include "KaleidoscopeDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "KaleidoscopeTypes.cpp.inc"

#define GET_OP_CLASSES
#include "KaleidoscopeOps.cpp.inc"
```

> **What's with the macros?** They're not configuration flags — they're instructions to the generated `.inc` file telling it which section you want right now. Each `#include` consumes the macro once. If that sounds strange, don't worry; just copy the pattern.

One more thing: generating classes doesn't automatically teach the `MLIRContext` about them. You need to register them when the dialect initializes:

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

Once `TheContext->loadDialect<kaleidoscope::KaleidoscopeDialect>()` runs, MLIR knows how to create, parse, print, and verify everything in your dialect.

## Step 5: Generate IR from Your AST

Now for the fun part — using the new operations. You'll update your AST generator to emit `kaleidoscope.*` operations instead of anonymous allocations.

**Creating a variable** becomes a single operation that captures everything at once:

```cpp
static Value CreateVariable(StringRef Name, Value InitialValue,
                            int64_t ArgumentNumber = 0) {
  return TheBuilder->create<kaleidoscope::DeclareOp>(
      getLocation(), kaleidoscope::VariableType::get(TheContext.get()),
      InitialValue, TheBuilder->getStringAttr(Name),
      TheBuilder->getI64IntegerAttr(ArgumentNumber));
}
```

**Reading a variable** becomes a `read`:

```cpp
return TheBuilder->create<kaleidoscope::ReadOp>(
    getLocation(), TheBuilder->getF64Type(), It->second);
```

**Assigning to a variable** becomes an `assign` (which, like before, also returns the assigned value so it can be used in expressions):

```cpp
TheBuilder->create<kaleidoscope::AssignOp>(getLocation(), It->second,
                                           AssignedValue);
return AssignedValue;
```

**Function arguments** use the same declaration, with a one-based argument number:

```cpp
unsigned Index = 0;
for (BlockArgument Argument : TheFunction.getArguments()) {
  StringRef Name = P.getArgs()[Index];
  Value Storage = CreateVariable(Name, Argument, Index + 1);
  NamedValues[Name.str()] = Storage;
  ++Index;
}
```

Locals and loop variables pass `0` instead.

**Bonus:** We can now delete the `FunctionParameters` map from Chapter 9. Each declaration carries its own name, location, and argument number, so there's nothing to reconstruct later.

## Step 6: Lower the Dialect to LLVM

When lowering time comes, we convert each `kaleidoscope.*` operation to LLVM IR.

**A declaration** becomes an allocation plus an initializing store:

```cpp
Value One = LLVM::ConstantOp::create(
    Rewriter, Loc, Rewriter.getI64Type(), Rewriter.getI64IntegerAttr(1));
Value Address = LLVM::AllocaOp::create(Rewriter, Loc, PointerType,
                                       DoubleType, One, 0);
LLVM::StoreOp::create(Rewriter, Loc, Adaptor.getInitialValue(), Address);
```

Here's the payoff: at this exact moment, we still have the source variable's name, location, and argument number in hand — *and* we've just produced its final stack address. That means we can create the debug declaration right here, without any reconstruction:

```cpp
auto Variable = LLVM::DILocalVariableAttr::get(
    Scope, Op.getName(), Scope.getFile(), Line,
    Op.getArgumentNumber(), /*alignInBits=*/0, VariableType,
    LLVM::DIFlags::Zero);
LLVM::DbgDeclareOp::create(
    Rewriter, Loc, Address, Variable,
    LLVM::DIExpressionAttr::get(Rewriter.getContext()));
```

**Reads and assignments** become simple loads and stores:

```cpp
Rewriter.replaceOpWithNewOp<LLVM::LoadOp>(
    Op, Rewriter.getF64Type(), Adaptor.getVariable());

Rewriter.replaceOpWithNewOp<LLVM::StoreOp>(
    Op, Adaptor.getValue(), Adaptor.getVariable());
```

Finally, tell MLIR that leaving a `kaleidoscope.*` operation unconverted is an error:

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

Without `addIllegalDialect`, a buggy pattern could silently skip an operation and pass IR with dangling `kaleidoscope` ops to the LLVM backend. Declaring the dialect illegal makes that failure loud.

## Step 7: Wire Up the Pass Pipeline

The order matters here. First, lower the standard high-level dialects down to LLVM. Then run our Chapter 9 debug pass to create the compile unit and function scopes:

```cpp
PassManager DebugPM(TheContext.get());
DebugPM.addPass(createKaleidoscopeDebugInfoPass(
    InputFilename.getValue(), OptLevel));
```

With function scopes now available, we can run the variable-lowering pass, which creates both the stack storage and the variable debug declarations together:

```cpp
DebugPM.addPass(std::make_unique<LowerKaleidoscopeVariablesPass>());
```

Then let MLIR's standard pass fill in the remaining scopes on the lowered operations:

```cpp
LLVM::DIScopeForLLVMFuncOpPassOptions DebugOptions;
DebugOptions.emissionKind = LLVM::DIEmissionKind::Full;
DebugPM.addPass(
    LLVM::createDIScopeForLLVMFuncOpPass(std::move(DebugOptions)));
```

The resulting LLVM IR looks like this:

```llvm
%x = alloca double, i64 1, align 8
store double %arg0, ptr %x, align 8
#dbg_declare(ptr %x, !variable, !DIExpression(), !location)
```

From here, everything downstream — translation, JIT, object emission, DWARF generation — proceeds unchanged.

## Step 8: Try It Out

Let's verify everything works. Run the interpreter:

```text
ready> def test(x) var y = x in (y = y + 1) * y;
Read function definition:
ready> test(4);
Evaluated to 25.000000
```

Now dump the MLIR before lowering to see your named variables:

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

And the LLVM IR after lowering, where variables have become stack allocations with debug declarations attached:

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

The first declaration describes parameter `x`; the second describes local `y`. Notice that `arg: 1` matches the `argumentNumber` we set on the declaration.

## Step 9: Run the Tests

Dialect conversion is exactly the kind of thing that deserves a regression test. This chapter includes a lit test that exercises all three variable operations and then runs the compiled result:

```bash
cmake --build build --target check-chapter-10
```

## Where the Code Lives

The full implementation is spread across these files:

- **`toy.cpp`** — the compiler and the variable-lowering pass
- **`KaleidoscopeOps.td`** — the variable type and operation definitions
- **`KaleidoscopeDialect.h` / `.cpp`** — connects generated classes to the compiler
- **`KaleidoscopeDebugInfo.h` / `.cpp`** — creates compile-unit and function scopes
- **`CMakeLists.txt`** — runs TableGen and builds the executable

### Build Configuration

```cmake(../code/chapter-10/CMakeLists.txt)
```

Build as usual:

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

## Wrapping Up

Our dialect is intentionally tiny — three operations and a type. That's the point.

We built it because a source variable knows things an anonymous allocation doesn't: its name, its location in the source, and its parameter number. By keeping that knowledge alive in the IR until the moment of lowering, we can generate storage *and* debug information together, instead of reconstructing the relationship after the fact.

That's the real reason to reach for a dialect: it lets your compiler hold onto the concepts that matter to *your* language, for exactly as long as it needs to, before expressing them in more general operations.

If you want to take this further, the next natural step would be moving the entire Kaleidoscope AST into a dialect — the [MLIR Toy tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/) shows what that architecture looks like. But for the problem we set out to solve, three operations were enough.