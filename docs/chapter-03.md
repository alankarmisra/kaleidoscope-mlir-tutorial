# 3. Kaleidoscope: Code generation to MLIR

## Chapter 3 Introduction

Welcome to Chapter 3 of the "[Implementing a language with
MLIR](chapter-00.md)" tutorial. This chapter shows you how to transform
the [Abstract Syntax Tree](chapter-02.md), built in Chapter 2, into
MLIR, and, in the next chapter, to LLVM IR. This will teach you a little bit about how MLIR does things, as
well as demonstrate how easy it is to use. It's much more work to build
a lexer and parser than it is to generate MLIR code. :)

**Please note**: the code in this chapter and later was written and
tested with MLIR from LLVM 21.1.6. MLIR's C++ APIs can change between
LLVM releases, so other versions may require changes. LLVM releases are
available from the [LLVM project releases page](https://llvm.org/releases/).

## Code Generation Setup

In order to generate MLIR, we want some simple setup to get started.
First we define virtual code generation (codegen) methods in each AST
class:

```cpp
/// ExprAST - Base class for all expression nodes.
class ExprAST {
public:
  virtual ~ExprAST() = default;
  virtual Value codegen() = 0;
};

/// NumberExprAST - Expression class for numeric literals like "1.0".
class NumberExprAST : public ExprAST {
  double Val;

public:
  NumberExprAST(double Val) : Val(Val) {}
  Value codegen() override;
};
...
```

The codegen() method says to emit IR for that AST node along with all the things it depends on, and they all return an MLIR Value object. "Value" is the class used to represent a "[Static Single Assignment (SSA)](http://en.wikipedia.org/wiki/Static_single_assignment_form) value" or "SSA value" in MLIR. The most distinct aspect of SSA values is that their value is computed as the related operation executes, and it does not get a new value until (and if) the operation re-executes. In other words, there is no way to "change" an SSA value. For more information, please read up on [Static Single Assignment](http://en.wikipedia.org/wiki/Static_single_assignment_form) - the concepts are really quite natural once you grok them.

Note that instead of adding virtual methods to the ExprAST class hierarchy, it could also make sense to use a [visitor pattern](http://en.wikipedia.org/wiki/Visitor_pattern) or some other way to model this. Again, this tutorial won't dwell on good software engineering practices: for our purposes, adding a virtual method is simplest.

The second thing we want is a `LogError` method like we used for the parser, which will be used to report errors found during code generation (for example, use of an undeclared parameter):

```cpp
static std::unique_ptr<MLIRContext> TheContext;
static OwningOpRef<ModuleOp> TheModule;
static std::unique_ptr<OpBuilder> TheBuilder;
static std::map<std::string, Value> NamedValues;

static Location getLocation() { return TheBuilder->getUnknownLoc(); }

Value LogErrorV(const char *Str) {
  LogError(Str);
  return {};
}
```

The static variables will be used during code generation. `TheContext` is an opaque object that owns a lot of core MLIR data structures, such as the type and attribute tables. We don't need to understand it in detail, we just need a single instance to pass into APIs that require it.

`TheBuilder` is a helper object that makes it easy to generate MLIR operations. Instances of the
[OpBuilder](https://mlir.llvm.org/doxygen/classmlir_1_1OpBuilder.html) class keep track of the current place to insert operations and have methods to create new operations.

`TheModule` is an MLIR construct that contains functions and top-level operations. In many ways, it is the top-level structure that the MLIR uses to contain code. It will own the memory for all of the IR that we generate, which is why the `codegen()` method returns a non-owning `Value` handle, rather than a `unique_ptr<Value>`.

The `NamedValues` map keeps track of which values are defined in the current scope and what their MLIR representation is. (In other words, it is a symbol table for the code). In this form of Kaleidoscope, the only things that can be referenced are function parameters. As such, function parameters will be in this map when generating code for their function
body.

Every MLIR operation has a location, which can provide a link back to the original source. For now, though, `getLocation()` simply returns an unknown location.

With these basics in place, we can start talking about how to generate code for each expression. Note that this assumes that `TheBuilder` has been set up to generate code *into* something. For now, we'll assume that this has already been done, and we'll just use it to emit code.

## Expression Code Generation

Generating MLIR code for expression nodes is very straightforward. First we'll do numeric literals:

```cpp
Value NumberExprAST::codegen() {
  return TheBuilder->create<arith::ConstantOp>(
      getLocation(), TheBuilder->getF64FloatAttr(Val));
}
```

In MLIR, one way to represent numeric constants is to use the
`arith::ConstantOp` operation from MLIR's
[`arith`](https://mlir.llvm.org/docs/Dialects/ArithOps/) *dialect* for
arithmetic operations. For now, you can think of dialects as libraries of related operations
and types. The dialects used in this tutorial provide convenient ways
to express arithmetic, functions, control flow, and memory at a higher
level, before MLIR progressively lowers them to LLVM IR. Where relevant,
we will compare these higher-level MLIR operations with the corresponding
LLVM IR to see how they simplify code generation. We will discuss
dialects in more detail and build a custom Kaleidoscope dialect in a
later chapter to represent language-specific operations and types that
existing dialects do not capture directly.

The call to `getF64FloatAttr(Val)` creates an `f64` floating-point attribute containing the numeric value. This code basically just creates and inserts a constant operation at the builder's current insertion point. The operation produces an SSA result, which is returned as a `Value`. MLIR attributes are uniqued and shared which is why the float attribute uses the `get` idiom. The constant operations that use those attributes are ordinary operations and are not themselves uniqued which is why the operation uses the `create` idiom.

```cpp
Value VariableExprAST::codegen() {
  // Look this variable up in the function.
  auto It = NamedValues.find(Name);
  if (It == NamedValues.end())
    return LogErrorV("Unknown variable name");
  return It->second;
}
```

References to variables are also quite simple using MLIR. In the simple
version of Kaleidoscope, we assume that the variable has already been
emitted somewhere and its value is available. In practice, the only
values that can be in the `NamedValues` map are function arguments.
This code simply checks to see that the specified name is in the map (if
not, an unknown variable is being referenced) and returns the value for
it. In future chapters, we'll add support for [loop induction
variables](chapter-05.md#for-loop-expression) in the symbol table, and for
[local variables](chapter-07.md#user-defined-local-variables).

```cpp
Value BinaryExprAST::codegen() {
  Value L = LHS->codegen();
  Value R = RHS->codegen();
  if (!L || !R)
    return {};

  switch (Op) {
  case '+':
    return TheBuilder->create<arith::AddFOp>(getLocation(), L, R);
  case '-':
    return TheBuilder->create<arith::SubFOp>(getLocation(), L, R);
  case '*':
    return TheBuilder->create<arith::MulFOp>(getLocation(), L, R);
  case '<': {
    Value Comparison = TheBuilder->create<arith::CmpFOp>(
        getLocation(), arith::CmpFPredicate::ULT, L, R);
    // Convert bool 0/1 to double 0.0 or 1.0.
    return TheBuilder->create<arith::UIToFPOp>(
        getLocation(), TheBuilder->getF64Type(), Comparison);
  }
  default:
    return LogErrorV("invalid binary operator");
  }
}
```

Binary operators start to get more interesting. The basic idea here is
that we recursively emit code for the left-hand side of the expression,
then the right-hand side, then we compute the result of the binary
expression. In this code, we do a simple switch on the opcode to create
the right MLIR `arith` operation.

In the example above, the MLIR builder class is starting to show its
value. OpBuilder knows where to insert the newly created operation,
all you have to do is specify what operation to create (e.g. with
`create<arith::AddFOp>`), which operands to use (`L` and `R` here).

MLIR automatically assigns each SSA value a unique textual name when
the IR is printed. These names exist only to make the printed IR
readable and are not stored as part of the value's identity. Internally,
MLIR represents each value as a handle to an operation result or block
argument and tracks its uses directly. Consequently, the printed names
may change when the IR is transformed or printed again.

[MLIR operations](https://mlir.llvm.org/docs/LangRef/#operations) are
constrained by strict rules. For example, the left and right operands of
an [`arith.addf`](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithaddf-arithaddfop)
operation must have the same type, and the result type must match the
operand types. Because all values in Kaleidoscope are doubles, this makes
for very simple code for add, sub, and mul.

On the other hand, the [`arith.cmpf`](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithcmpf-arithcmpfop)
operation returns an `i1` value when comparing scalar operands. The problem with this is that Kaleidoscope wants the value to be a `0.0` or `1.0`. To get these semantics, we combine `arith.cmpf` with an [`arith.uitofp`](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithuitofp-arithuitofpop) operation. This operation converts its input integer into a floating-point value by treating the input as unsigned. In contrast, if we used an [`arith.sitofp`](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithsitofp-arithsitofpop) operation, the Kaleidoscope `<` operator would return `0.0` or `-1.0`, depending on the comparison result.

```cpp
Value CallExprAST::codegen() {
  // Look up the name in the global module table.
  auto CalleeF = TheModule->lookupSymbol<func::FuncOp>(Callee);
  if (!CalleeF)
    return LogErrorV("Unknown function referenced");

  // If argument mismatch error.
  if (CalleeF.getNumArguments() != Args.size())
    return LogErrorV("Incorrect # arguments passed");

  std::vector<Value> ArgsV;
  for (auto &Arg : Args) {
    ArgsV.push_back(Arg->codegen());
    if (!ArgsV.back())
      return {};
  }

  return TheBuilder->create<func::CallOp>(getLocation(), CalleeF, ArgsV)
      .getResult(0);
}
```

Code generation for function calls is quite straightforward with MLIR and the [func](https://mlir.llvm.org/docs/Dialects/Func/) dialect (yes there's a dialect for most common operations - which is what makes MLIR so useful!). The code above initially does a function name lookup in the MLIR Module's symbol table. Recall that the MLIR Module is the container that holds the functions we are JIT'ing. By giving each function the same name as what the user specifies, we can use the MLIR symbol table to resolve function names for us.

Once we have the function to call, we recursively codegen each argument that is to be passed in, and create an MLIR [`func.call`](https://mlir.llvm.org/docs/Dialects/Func/#funccall-funccallop) Operation. In the next chapter, we'll see how these calls are lowered using the default C calling convention, allowing us to call external C functions like `sin` and `cos`.

This wraps up our handling of the four basic expressions that we have so far in Kaleidoscope. Feel free to go in and add some more. For example, by browsing the [arith dialect](https://mlir.llvm.org/docs/Dialects/ArithOps/) you'll find
several other interesting operations that are really easy to plug into our basic framework.

## Function Code Generation

Code generation for prototypes and functions must handle a number of
details, which make their code less beautiful than expression code
generation, but allows us to illustrate some important points. First,
let's talk about code generation for prototypes: they are used both for
function bodies and external function declarations. The code starts
with:

```cpp
func::FuncOp PrototypeAST::codegen() {
  // Make the function type: double(double, double), etc.
  std::vector<Type> Doubles(Args.size(), TheBuilder->getF64Type());
  auto FunctionType =
      TheBuilder->getFunctionType(Doubles, {TheBuilder->getF64Type()});

  auto Function = func::FuncOp::create(getLocation(), Name, FunctionType);
  TheModule->push_back(Function);
  return Function;
}
```

This code packs a lot of power into a few lines. Note first that this
function returns a `func::FuncOp` instead of a `Value`. Because a
“prototype” really talks about the external interface for a function
(not the value computed by an expression), it makes sense for it to
return the MLIR function operation it corresponds to when codegen'd.

The call to `getFunctionType` creates the `FunctionType` that should be
used for a given prototype. Since all function arguments in Kaleidoscope
are of type double, the first line creates a vector of “N” MLIR `f64`
types. It then uses `getFunctionType` to create a function type that
takes “N” `f64` values as arguments and returns one `f64` value as its
result. Note that types in MLIR are uniqued, so you don't “new” a type;
you ask the builder or context to “get” it.

The call to `func::FuncOp::create` creates the IR function corresponding
to the prototype. This specifies the function's location, name, and
type. The call to `TheModule->push_back` then inserts the function into
the module. A `func.func` operation is also an MLIR
[symbol](https://mlir.llvm.org/docs/SymbolsAndSymbolTables/), so its name
is registered in `TheModule`'s symbol table when it is inserted.

At this point we have a function prototype with no body. This is how `func.func`
represents function declarations. For extern statements in Kaleidoscope, this
is as far as we need to go. For function definitions however, we need to
codegen and attach a function body.

```cpp
func::FuncOp FunctionAST::codegen() {
  // First, check for an existing function from a previous 'extern' declaration.
  auto TheFunction = TheModule->lookupSymbol<func::FuncOp>(Proto->getName());

  if (!TheFunction)
    TheFunction = Proto->codegen();

  if (!TheFunction)
    return {};

  if (!TheFunction.isDeclaration()) {
    LogError("Function cannot be redefined.");
    return {};
  }
```

For function definitions, we start by searching TheModule's symbol table for an existing version of this function, in case one has already been created using an 'extern' statement. If `TheModule->lookupSymbol` returns null then no previous version exists, so we'll codegen one from the Prototype. In either case, we want to assert that the function is empty (i.e. has no body yet) before we start. `TheFunction.isDeclaration()` returns true if the body is empty.  

```cpp
  // Create a new basic block to start insertion into.
  Block *EntryBlock = TheFunction.addEntryBlock();
  TheBuilder->setInsertionPointToStart(EntryBlock);

  // Record the function arguments in the NamedValues map.
  NamedValues.clear();
  unsigned Index = 0;
  for (BlockArgument Argument : TheFunction.getArguments())
    NamedValues[Proto->getArgs()[Index++]] = Argument;
```

Now we get to the point where `TheBuilder` is set up. The first line
adds a new [basic block](http://en.wikipedia.org/wiki/Basic_block) to
`TheFunction`. The second line then tells the builder that new
operations should be inserted at the start of the new basic block. Blocks
in MLIR are an important part of regions and define the
[Control Flow Graph](http://en.wikipedia.org/wiki/Control_flow_graph).
Since we don't have any control flow, our functions will only contain
one block at this point. We'll fix this in [Chapter 5](chapter-05.md) :).

Next we add the function arguments to the `NamedValues` map (after first clearing
it out) so that they're accessible to `VariableExprAST` nodes.

```cpp
  if (Value RetVal = Body->codegen()) {
    // Finish off the function.
    TheBuilder->create<func::ReturnOp>(getLocation(), RetVal);

    // Validate the generated code, checking for consistency.
    if (succeeded(verify(TheFunction)))
      return TheFunction;
  }
```

Once the insertion point has been set up and the NamedValues map populated, we call the `codegen()` method for the root expression of the function. If no error happens, this emits code to compute the expression into the entry block and returns the value that was computed. Assuming no error, we then create a [func.return](https://mlir.llvm.org/docs/Dialects/Func/#funcreturn-funcreturnop) operation, which completes the function.

Once the function is built, we call `verify`, which is
provided by MLIR. This function does a variety of consistency checks on
the generated operations, to determine if our compiler is doing everything
right. Using this is important: it can catch a lot of bugs. Once the
function is finished and validated, we return it.

```cpp
  // Error reading body, remove function.
  TheFunction.erase();
  return {};
}
```

The only piece left here is handling of the error case. For simplicity,
we handle this by merely deleting the function we produced with the
`erase` method. This allows the user to redefine a function
that they incorrectly typed in before: if we didn't delete it, it would
live in the symbol table, with a body, preventing future redefinition.

This code does have a bug, though: if `FunctionAST::codegen()` finds an
existing MLIR function, it does not validate its type against the
definition's own prototype. This means that an earlier `extern`
declaration takes precedence over the function definition's signature.
Because all Kaleidoscope values currently have type `f64`, the relevant
difference is the number of arguments. There are a number of ways to fix
this bug; see what you can come up with! Here is a testcase (the `--dump-mlir` option is explained in a bit):

<!-- code-merge:start -->
```bash
$ build/toy --dump-mlir
```
```kaleidoscope
ready> extern foo(a);
```
```text
Read extern:
```
```mlir
func.func private @foo(f64) -> f64
```
```kaleidoscope
ready> def foo(a b) a;
```
```text
Read function definition:
```
```mlir
func.func private @foo(%arg0: f64) -> f64 {
  return %arg0 : f64
}
```
<!-- code-merge:end -->

The MLIR verifier cannot catch this because both prototypes are parsed, but
`FunctionAST::codegen()` finds the existing one-argument `@foo` created by the
`extern` and reuses it without checking it against the two-argument definition.
The resulting module therefore contains one internally valid function—but it
is the wrong one-argument function.

## Private declarations

```cpp
static void HandleExtern() {
  if (auto ProtoAST = ParseExtern()) {
    if (auto FnIR = ProtoAST->codegen()) {
      // Function declarations need to be private
      FnIR.setPrivate();
      fprintf(stderr, "Read extern:\n");
      ...
    }
  } else {
    // Skip token for error recovery.
    getNextToken();
  }
}
```

MLIR allows function definitions with bodies to have public visibility,
making their symbols available outside the module. Function declarations
have no body and represent functions supplied externally, so the `func`
dialect requires them to have private symbol visibility. This allows
operations within the module to reference a declaration such as `cos`
without treating it as a definition exported by the module. The actual
`cos` function is resolved later by the JIT or linker.

## The `--dump-mlir` Option

By default, the compiler does not print the generated MLIR. To see the IR,
run Kaleidoscope with the `--dump-mlir` option:

<!-- code-merge:start -->
```bash
$ build/toy --dump-mlir
```
```kaleidoscope
ready> def add(a b) a + b;
```
```text
Read function definition:
```
```mlir
func.func @add(%arg0: f64, %arg1: f64) -> f64 {
  %0 = arith.addf %arg0, %arg1 : f64
  return %0 : f64
}
```
<!-- code-merge:end -->

The option is defined using LLVM's command-line support:

```cpp
static llvm::cl::opt<bool> DumpMLIR(
    "dump-mlir", llvm::cl::desc("Print generated MLIR"),
    llvm::cl::init(false));
```

After generating a function, the driver checks the option and prints the
corresponding MLIR operation:

```cpp
if (DumpMLIR) {
  llvm::errs() << "Read function definition:\n";
  FnIR.print(llvm::errs(), OpPrintingFlags().assumeVerified());
  llvm::errs() << '\n';
}
```

## Driver Changes and Closing Thoughts

For now, code generation to MLIR doesn't really get us much, except that
we can look at the pretty IR. The sample code inserts calls to
codegen into the "`HandleDefinition`", "`HandleExtern`" etc
functions, and then dumps out MLIR. This gives a nice way to look
at the MLIR for simple functions. For example:

<!-- code-merge:start -->
```text
$ build/toy --dump-mlir
```
```mlir
ready> 4+5;
Read top-level expression:
func.func @__anon_expr() -> f64 {
  %cst = arith.constant 4.000000e+00 : f64
  %cst_0 = arith.constant 5.000000e+00 : f64
  %0 = arith.addf %cst, %cst_0 : f64
  return %0 : f64
}
```
<!-- code-merge:end -->

Note how the parser turns the top-level expression into anonymous
functions for us. This will be handy when we add [JIT
support](chapter-04.md#adding-a-jit-compiler) in the next chapter. Also note that the
code is very literally transcribed, no optimizations are being performed. We will [add
optimizations](chapter-04.md#why-we-need-an-optimization-pipeline) explicitly in the next
chapter.

```mlir
ready> def foo(a b) a*a + 2*a*b + b*b;
Read function definition:
func.func @foo(%arg0: f64, %arg1: f64) -> f64 {
  %0 = arith.mulf %arg0, %arg0 : f64
  %cst = arith.constant 2.000000e+00 : f64
  %1 = arith.mulf %cst, %arg0 : f64
  %2 = arith.mulf %1, %arg1 : f64
  %3 = arith.addf %0, %2 : f64
  %4 = arith.mulf %arg1, %arg1 : f64
  %5 = arith.addf %3, %4 : f64
  return %5 : f64
}
```

This shows some simple arithmetic. Notice the striking similarity to the
MLIR builder calls that we use to create the operations.

```mlir
ready> def bar(a) foo(a, 4.0) + bar(31337);
Read function definition:
func.func @bar(%arg0: f64) -> f64 {
  %cst = arith.constant 4.000000e+00 : f64
  %0 = call @foo(%arg0, %cst) : (f64, f64) -> f64
  %cst_0 = arith.constant 3.133700e+04 : f64
  %1 = call @bar(%cst_0) : (f64) -> f64
  %2 = arith.addf %0, %1 : f64
  return %2 : f64
}
```

This shows some function calls. Note that this function will take a long
time to execute if you call it. In the future we'll add conditional
control flow to actually make recursion useful :).

```mlir
ready> extern cos(x);
Read extern:
func.func private @cos(f64) -> f64

ready> cos(1.234);
Read top-level expression:
func.func @__anon_expr() -> f64 {
  %cst = arith.constant 1.234000e+00 : f64
  %0 = call @cos(%cst) : (f64) -> f64
  return %0 : f64
}
```

This shows an extern for the libm "cos" function, and a call to it.

```mlir
ready> ^D
module {
  func.func @foo(%arg0: f64, %arg1: f64) -> f64 {
    %0 = arith.mulf %arg0, %arg0 : f64
    %cst = arith.constant 2.000000e+00 : f64
    %1 = arith.mulf %cst, %arg0 : f64
    %2 = arith.mulf %1, %arg1 : f64
    %3 = arith.addf %0, %2 : f64
    %4 = arith.mulf %arg1, %arg1 : f64
    %5 = arith.addf %3, %4 : f64
    return %5 : f64
  }
  func.func @bar(%arg0: f64) -> f64 {
    %cst = arith.constant 4.000000e+00 : f64
    %0 = call @foo(%arg0, %cst) : (f64, f64) -> f64
    %cst_0 = arith.constant 3.133700e+04 : f64
    %1 = call @bar(%cst_0) : (f64) -> f64
    %2 = arith.addf %0, %1 : f64
    return %2 : f64
  }
  func.func private @cos(f64) -> f64
}
```

When you quit the current demo by sending an EOF via CTRL+D on Linux or
macOS, or CTRL+Z and ENTER on Windows, it dumps the MLIR for the complete
module. Here you can see the larger structure and how its functions
reference one another. Top-level expressions do not appear in this final
module because the driver erases each anonymous function after printing it.

This wraps up the third chapter of the Kaleidoscope tutorial. Up next,
we'll describe how to [add JIT codegen and optimizer
support](chapter-04.md) to this so we can actually start running
code!

## Full Code Listing

Here is the complete code listing for our running example, enhanced with
the MLIR code generator. Because this uses the MLIR libraries, you need
to build MLIR before compiling it. See the
[MLIR getting started guide](https://mlir.llvm.org/getting_started/) for
instructions.

We use the following `CMakeLists.txt` to build the example:

```cmake(../code/chapter-03/CMakeLists.txt)
```

Configure the example by setting `MLIR_DIR` to the directory containing
`MLIRConfig.cmake` in your LLVM build:

```bash
cmake -S . -B build \
  -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir

cmake --build build

# Run
./build/toy
```

Here is the code:

```cpp(../code/chapter-03/toy.cpp)
```

[Next: Adding JIT and Optimizer Support](chapter-04.md)
