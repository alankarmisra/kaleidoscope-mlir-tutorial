# 10. Kaleidoscope: Building a Custom Dialect

## Chapter 10 Introduction

Welcome to Chapter 10 of the "[Implementing a language with
MLIR](chapter-00.md)" tutorial. In the previous chapters, we generated
operations from MLIR's existing dialects directly from our AST. This has
served us well: `arith` represents our arithmetic, `func` represents our
functions, `scf` represents our control flow, and `memref` provides storage
for mutable variables.

There are times, however, when the operations provided by existing dialects
do not describe everything that the source language knows. Lowering that
information immediately can make language-specific analysis and
transformation more difficult. MLIR allows a language to define its own
*dialect* and preserve these concepts until they are no longer useful.

In this chapter, we'll add a small Kaleidoscope dialect containing one
operation for binary operators. We won't change the Kaleidoscope language at
all. Instead, we will change the initial MLIR it produces and add a pass that
progressively lowers our new operation into the dialects we already use.

## Why Preserve Binary Operators?

Consider this function:

```kaleidoscope
def arithmetic(x y)
  (x + y) * y;
```

Until now, `BinaryExprAST::codegen()` immediately selected an implementation
for each operator. The `+` became `arith.addf`, `*` became `arith.mulf`, `<`
became a comparison followed by a conversion, and a user-defined operator
became a `func.call`.

In this chapter, the initial MLIR looks like this:

```mlir
func.func @arithmetic(%arg0: f64, %arg1: f64) -> f64 {
  // Loads from the mutable argument slots have been omitted here.
  %0 = kaleidoscope.binary "+" %x, %y : f64
  %1 = kaleidoscope.binary "*" %0, %y : f64
  return %1 : f64
}
```

This IR records what the source program said: these values were produced by
Kaleidoscope binary operators. It does not yet decide how those operators
will be implemented. That decision moves out of AST generation and into an
explicit dialect-conversion pass.

This is the important idea behind progressive lowering. A compiler can keep
an operation at the level where it is most useful, perform any analysis or
transformation appropriate at that level, and only then replace it with more
general operations.

## Defining the Dialect and Operation

MLIR operations are commonly defined using the
[Operation Definition Specification](https://mlir.llvm.org/docs/DefiningDialects/Operations/),
or ODS. ODS uses TableGen records to describe an operation's name, operands,
results, attributes, traits, assembly form, and generated C++ API.

We begin by declaring our dialect and a base class for its operations:

```tablegen
def Kaleidoscope_Dialect : Dialect {
  let name = "kaleidoscope";
  let cppNamespace = "::mlir::kaleidoscope";
  let summary = "Operations that preserve Kaleidoscope language semantics";
}

class Kaleidoscope_Op<string mnemonic, list<Trait> traits = []>
    : Op<Kaleidoscope_Dialect, mnemonic, traits>;
```

The dialect name provides the `kaleidoscope.` prefix in textual MLIR. The C++
namespace keeps the generated classes separate from other dialects.

Our binary operation is also defined in TableGen:

```tablegen
def Kaleidoscope_BinaryOp : Kaleidoscope_Op<"binary", [
    SameOperandsAndResultType
  ]> {
  let summary = "a built-in or user-defined Kaleidoscope binary operator";
  let arguments = (ins F64:$lhs, F64:$rhs, StrAttr:$operatorName);
  let results = (outs F64:$result);

  let builders = [
    OpBuilder<(ins "StringRef":$operatorName, "Value":$lhs,
                   "Value":$rhs), [{
      build($_builder, $_state, lhs.getType(), lhs, rhs,
            $_builder.getStringAttr(operatorName));
    }]>
  ];

  let assemblyFormat = "$operatorName $lhs `,` $rhs attr-dict `:` type($result)";
  let hasVerifier = 1;
}
```

`BinaryOp` has two `f64` operands, one `f64` result, and a string attribute
containing the operator character. `SameOperandsAndResultType` records that
its operands and result have the same type. We deliberately do not mark the
operation `Pure`: a user-defined operator may call a function with side
effects, even though Kaleidoscope's built-in arithmetic operators are pure.

The assembly format gives us the compact representation used above:

```mlir
%0 = kaleidoscope.binary "+" %lhs, %rhs : f64
```

We also request a custom verifier. Kaleidoscope operators contain exactly one
character, and assignment is handled separately because it stores into a
mutable variable:

```cpp
LogicalResult BinaryOp::verify() {
  StringRef Operator = getOperatorName();
  if (Operator.size() != 1)
    return emitOpError("requires a one-character operator");
  if (Operator == "=")
    return emitOpError("does not represent the assignment operator");
  return success();
}
```

An operation may also provide folding behavior. Our folder handles the narrow
case where both operands are floating-point constants:

```cpp
OpFoldResult BinaryOp::fold(FoldAdaptor Adaptor) {
  auto LHS = dyn_cast_or_null<FloatAttr>(Adaptor.getLhs());
  auto RHS = dyn_cast_or_null<FloatAttr>(Adaptor.getRhs());
  if (!LHS || !RHS)
    return {};

  APFloat Result = LHS.getValue();
  StringRef Operator = getOperatorName();
  if (Operator == "+")
    Result.add(RHS.getValue(), APFloat::rmNearestTiesToEven);
  else if (Operator == "-")
    Result.subtract(RHS.getValue(), APFloat::rmNearestTiesToEven);
  else if (Operator == "*")
    Result.multiply(RHS.getValue(), APFloat::rmNearestTiesToEven);
  else
    return {};

  return FloatAttr::get(getResult().getType(), Result);
}
```

This is an example of behavior that can travel with an operation while it is
part of the Kaleidoscope dialect. It is not something that requires a custom
dialect: a compiler could also register patterns for `arith` operations. The
reason for keeping `kaleidoscope.binary` is the separation between recording
the source-language operation and deciding how to implement it.

When a folder returns an attribute, MLIR asks the dialect to materialize that
attribute as an operation. Our dialect delegates this job to
`arith::ConstantOp`, since `arith` already has the constant operation we need:

```cpp
Operation *KaleidoscopeDialect::materializeConstant(
    OpBuilder &Builder, Attribute Value, Type Type, Location Loc) {
  return Builder.create<arith::ConstantOp>(
      Loc, Type, cast<TypedAttr>(Value));
}
```

For example, `2 + 3` is folded into an `arith.constant` containing `5.0`.

TableGen generates most of the C++ operation class for us. The dialect's
`initialize()` method registers the generated operation:

```cpp
void KaleidoscopeDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "KaleidoscopeOps.cpp.inc"
      >();
}
```

Finally, the generated files are added to the build using `mlir_tablegen`.
The complete commands are available in the `CMakeLists.txt` for this chapter.

## Generating Kaleidoscope IR

The language grammar and AST remain unchanged. Assignment still needs its
special handling, and we still recursively generate the left- and right-hand
sides. Once those values are available, every other binary expression now
creates the same operation:

```cpp
Value L = LHS->codegen();
Value R = RHS->codegen();
if (!L || !R)
  return {};

StringRef Operator(&Op, 1);
if (Operator != "+" && Operator != "-" && Operator != "*" &&
    Operator != "<" && !getFunction("binary" + Operator.str()))
  return LogErrorV("Unknown binary operator");

return TheBuilder->create<kaleidoscope::BinaryOp>(
    getLocation(), Operator, L, R);
```

We retain the lookup for user-defined operators. Besides reporting an unknown
operator promptly, JIT mode uses this lookup to reproduce its function
declaration in the current module. The operation itself remains a
`kaleidoscope.binary` until the lowering pass runs.

For example, the language we added in Chapter 6 continues to work unchanged:

```kaleidoscope
def binary % 40 (x y)
  x * y;

def custom(x y)
  x % y;
```

The second function initially contains:

```mlir
%0 = kaleidoscope.binary "%" %x, %y : f64
```

## Lowering the Kaleidoscope Dialect

Now we need to describe how `kaleidoscope.binary` is implemented. An
`OpConversionPattern` matches our operation and replaces it with legal
operations from other dialects.

The built-in arithmetic operators are straightforward:

```cpp
if (Operator == "+") {
  Rewriter.replaceOpWithNewOp<arith::AddFOp>(Op, LHS, RHS);
  return success();
}
if (Operator == "-") {
  Rewriter.replaceOpWithNewOp<arith::SubFOp>(Op, LHS, RHS);
  return success();
}
if (Operator == "*") {
  Rewriter.replaceOpWithNewOp<arith::MulFOp>(Op, LHS, RHS);
  return success();
}
```

The `<` operator keeps Kaleidoscope's `0.0` or `1.0` result semantics:

```cpp
if (Operator == "<") {
  Value Comparison = Rewriter.create<arith::CmpFOp>(
      Loc, arith::CmpFPredicate::ULT, LHS, RHS);
  Rewriter.replaceOpWithNewOp<arith::UIToFPOp>(
      Op, Rewriter.getF64Type(), Comparison);
  return success();
}
```

Anything else is a user-defined operator. Its implementation is the function
whose name begins with `binary`, just as it was in previous chapters:

```cpp
std::string FunctionName = "binary" + Operator.str();
auto Module = Op->getParentOfType<ModuleOp>();
auto Function = Module.lookupSymbol<func::FuncOp>(FunctionName);
if (!Function)
  return Rewriter.notifyMatchFailure(
      Op, "unknown user-defined operator");

Rewriter.replaceOpWithNewOp<func::CallOp>(
    Op, Function, ValueRange{LHS, RHS});
return success();
```

The pass declares the Kaleidoscope dialect illegal and supplies the pattern
that knows how to eliminate it:

```cpp
ConversionTarget Target(Context);
Target.addIllegalDialect<kaleidoscope::KaleidoscopeDialect>();
Target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

RewritePatternSet Patterns(&Context);
Patterns.add<BinaryOpLowering>(&Context);
if (failed(applyPartialConversion(
        getOperation(), Target, std::move(Patterns))))
  signalPassFailure();
```

Marking the dialect illegal is useful: lowering cannot silently succeed while
a `kaleidoscope.binary` operation remains in the module.

## Connecting the Lowering Pipeline

The new pass runs first in `lowerToLLVM()`:

```cpp
PassManager LoweringPM(TheContext.get());
LoweringPM.addPass(std::make_unique<LowerKaleidoscopePass>());
LoweringPM.addPass(createSCFToControlFlowPass());
LoweringPM.addPass(createConvertFuncToLLVMPass());
LoweringPM.addPass(createArithToLLVMConversionPass());
LoweringPM.addPass(createFinalizeMemRefToLLVMConversionPass());
LoweringPM.addPass(createConvertControlFlowToLLVMPass());
```

After the first pass, the module contains only the standard dialects already
handled by the rest of our pipeline. No changes are needed in the LLVM IR
translation, JIT, object-file emitter, or debug information support.

Running our earlier example shows the custom operation before lowering:

```mlir
%0 = kaleidoscope.binary "+" %x, %y : f64
%1 = kaleidoscope.binary "*" %0, %y : f64
```

The lowered LLVM IR contains the expected instructions:

```llvm
%sum = fadd double %x, %y
%result = fmul double %sum, %y
```

The same source can still be executed by the JIT:

```text
ready> arithmetic(2, 3);
Evaluated to 15.000000
```

and the alternative compilation mode still emits an object file:

```bash
./toy --emit-object < program.ks
```

## Testing the Dialect

Dialect conversion is a particularly good place for small regression tests.
This chapter includes a lit test that checks the initial custom operations and
then executes both built-in and user-defined operators through the JIT:

```bash
cmake --build build --target check-chapter-10
```

The test checks that built-in expressions produce
`kaleidoscope.binary`, that a user-defined `%` operator is represented by the
same operation, and that both paths evaluate to the expected results.

## Full Code Listing

The complete implementation is split across:

- `toy.cpp`, containing the compiler and lowering pass;
- `KaleidoscopeOps.td`, containing the ODS definitions;
- `KaleidoscopeDialect.h` and `KaleidoscopeDialect.cpp`, connecting the
generated operation classes to the compiler; and
- `CMakeLists.txt`, running TableGen and building the executable.

Use the chapter's build script as before:

```bash
./build.sh
```

## Closing Thoughts

We have not added any new Kaleidoscope syntax in this chapter. Instead, we
changed the level of abstraction used by the compiler. The AST now records a
source-language operation in a source-language dialect, and a separate pass
decides how that operation is implemented.

Our custom dialect is deliberately small, but it demonstrates the same
architecture used by much larger MLIR-based compilers: define operations that
preserve useful language or domain semantics, transform them while that
information is available, and progressively lower them into more general
dialects.

The [MLIR Toy tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/) takes this
idea further. Where Kaleidoscope preserves operator semantics, Toy preserves
tensor operations and shape information so that it can perform higher-level
analysis and transformation before lowering. With the custom-dialect workflow
from this chapter in place, you now have the background needed to continue
  there after finishing our [conclusion](chapter-11.md).
