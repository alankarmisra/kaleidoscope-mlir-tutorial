# 4. Kaleidoscope: Adding JIT and Optimizer Support

## Chapter 4 Introduction

Welcome to Chapter 4 of the "[Implementing a language with MLIR](chapter-00.md)" tutorial. Chapters 1-3 described the implementation
of a simple language and added support for generating MLIR. This
chapter describes two new techniques: adding optimizer support to your
language, and adding JIT compiler support. These additions will
demonstrate how to get nice, efficient code for the Kaleidoscope
language.

## Why We Need an Optimization Pipeline

Our demonstration for Chapter 3 is elegant and easy to extend.
Unfortunately, it does not produce wonderful code.

<!-- code-merge:start -->
```bash
$ ../chapter-03/build/toy --dump-mlir
```
```mlir
ready> def squareSumUnoptimized(x) (1+2+x)*(x+(1+2));
Read function definition:
func.func @squareSumUnoptimized(%arg0: f64) -> f64 {
  %cst = arith.constant 1.000000e+00 : f64
  %cst_0 = arith.constant 2.000000e+00 : f64
  %0 = arith.addf %cst, %cst_0 : f64
  %1 = arith.addf %0, %arg0 : f64
  %cst_1 = arith.constant 1.000000e+00 : f64
  %cst_2 = arith.constant 2.000000e+00 : f64
  %2 = arith.addf %cst_1, %cst_2 : f64
  %3 = arith.addf %arg0, %2 : f64
  %4 = arith.mulf %1, %3 : f64
  return %4 : f64
}
```
<!-- code-merge:end -->

In this case, we could have trivially folded `1 + 2` into `3`. Furthermore, the LHS and RHS of the multiplication compute the same value. We'd really like to see this generate `tmp = x + 3; result = tmp * tmp;`.

While constant folding can be achieved by examining each operation locally, no amount of such operation-local analysis will be able to detect and correct the duplicate computation of `x + 3`. This requires two transformations: canonicalization to make the additions identical and [Common Subexpression Elimination](https://en.wikipedia.org/wiki/Common_subexpression_elimination) (CSE) to delete the redundant add operation. Fortunately, MLIR provides a broad range of optimizations that you can use, in the form of "passes".

## MLIR Optimization Passes

MLIR provides many optimization passes, which do many different sorts of
things and have different tradeoffs. Unlike other systems, MLIR doesn't
hold to the mistaken notion that one set of optimizations is right for
all languages and for all situations. MLIR allows a compiler implementor
to make complete decisions about what optimizations to use, in which
order, and in what situation.

As a concrete example, MLIR passes can be anchored on different kinds of
operations. A pass anchored on `builtin.module` can examine and transform
the entire module, while a pass anchored on `func.func` operates on one
function at a time. This allows compiler implementors to choose the scope
at which each transformation runs. For more information on passes and how
they are run, see the
[MLIR Pass Infrastructure](https://mlir.llvm.org/docs/PassManagement/)
documentation and the
[MLIR Passes](https://mlir.llvm.org/docs/Passes/)
reference.

For Kaleidoscope, we are currently generating functions on the fly, one
at a time, as the user types them in. We aren't shooting for the
ultimate optimization experience in this setting, but we also want to
catch the easy and quick stuff where possible. As such, we will choose
to run a few per-function optimizations as the user types the function
in. A static Kaleidoscope compiler could take a simpler approach: generate
all functions into one MLIR module, run an optimization pipeline over the
completed module, lower it once, and emit the resulting object file.

In addition to the distinction between function and module passes, MLIR distinguishes transformation passes from analyses. Transformation passes mutate the IR. Analyses are read-only computations that transformations can request. MLIR computes and caches analyses on demand, invalidating them when the IR changes.

In order to get per-function optimizations going, we need to set up an
MLIR [PassManager](https://mlir.llvm.org/docs/PassManagement/) to hold
and organize the optimizations that we want to run. Once we have that,
we can add a set of optimizations nested under `func.func`. We'll need a
new `PassManager` for each module that we want to optimize, so we'll add
it to the function created in the previous chapter
(`InitializeModule()`):

```cpp
static void InitializeModuleAndManagers() {
  // Destroy objects that refer to the old context before replacing it.
  ThePM.reset();
  TheBuilder.reset();
  TheModule = OwningOpRef<ModuleOp>();
  TheContext.reset();

  // Open a new context and module.
  TheContext = std::make_unique<MLIRContext>();
  TheContext->loadDialect<arith::ArithDialect, func::FuncDialect>();
  TheModule = ModuleOp::create(UnknownLoc::get(TheContext.get()));

  // Create a new builder for the module.
  TheBuilder = std::make_unique<OpBuilder>(TheContext.get());

  // Create a pass manager and add a couple of simple optimizations.
  ThePM = std::make_unique<PassManager>(TheContext.get());
```

Once the pass manager is set up, we use a series of `addNestedPass` calls to add a couple of MLIR transformation passes:

```cpp
  ThePM->addNestedPass<func::FuncOp>(createCanonicalizerPass());
  ThePM->addNestedPass<func::FuncOp>(createCSEPass());
}
```

In this case, we choose to add two optimization passes. The passes we choose here are a pretty standard set of "cleanup" optimizations that are useful for a wide variety of code. I won't delve into what they do but, believe me, they are a good starting place :).

Once the `PassManager` is set up, we need to make use of it. We do this by
running it after our newly created function is constructed (in
`FunctionAST::codegen()`), but before it is returned to the client:

```cpp
if (Value RetVal = Body->codegen()) {
    // Finish off the function.
    TheBuilder->create<func::ReturnOp>(getLocation(), RetVal);

    // Validate the generated code, checking for consistency.
    if (succeeded(verify(TheFunction))) {
      // Run the optimizer on the module.
      if (failed(ThePM->run(*TheModule))) {
        LogError("Could not optimize function.");
        TheFunction.erase();
        return {};
      }
      return TheFunction;
    }
  }
```

As you can see, this is pretty straightforward. The `PassManager` runs the nested passes on each MLIR `func.func` operation in
place, improving (hopefully) its body. With this in place, we can try our test above again:

```mlir
ready> def squareSum(x) (1+2+x)*(x+(1+2));
Read function definition:
func.func @squareSum(%arg0: f64) -> f64 {
  %cst = arith.constant 3.000000e+00 : f64
  %0 = arith.addf %arg0, %cst : f64
  %1 = arith.mulf %0, %0 : f64
  return %1 : f64
}
```

As expected, we now get our nicely optimized code, saving a floating-point add operation from every execution of this function.

MLIR provides a wide variety of optimizations that can be used in certain circumstances. [Documentation about the various passes](https://mlir.llvm.org/docs/Passes/) is available. Another good source of ideas is to look at the pass pipelines used by other MLIR-based compilers. The [mlir-opt](https://mlir.llvm.org/docs/Tutorials/MlirOpt/) tool allows you to experiment with passes from the command line, so you can see what they do.

Now that we have reasonable code coming out of our front-end, let's talk about executing it!

## Adding a JIT Compiler

Code that is available in MLIR can have a wide variety of tools applied to it. For example, you can run optimizations on it (as we did above), you can dump it out in textual or binary forms, you can compile the code to an assembly file (.s) for some target, or you can JIT compile it. The nice thing about the MLIR representation is that it is the "common currency" between many different parts of the compiler.

In this section, we'll add JIT compiler support to our interpreter. The basic idea that we want for Kaleidoscope is to have the user enter function bodies as they do now, but immediately evaluate the top-level expressions they type in. For example, if they type in "1 + 2;", we should evaluate and print out 3. If they define a function, they should be able to call it from the command line.

In order to do this, we first prepare the environment to create code for the current native target and declare and initialize the JIT. This is done by calling some `InitializeNativeTarget*` functions and adding a global variable `TheJIT`, and initializing it in
`main`:

```cpp
static std::unique_ptr<llvm::orc::KaleidoscopeJIT> TheJIT;
...
int main() {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  // Install standard binary operators.
  // 1 is lowest precedence.
  BinopPrecedence['<'] = 10;
  BinopPrecedence['+'] = 20;
  BinopPrecedence['-'] = 20;
  BinopPrecedence['*'] = 40; // highest.

  // Prime the first token.
  fprintf(stderr, "ready> ");
  getNextToken();

  TheJIT = ExitOnErr(llvm::orc::KaleidoscopeJIT::Create());

  // Make the first module, which holds newly generated code.
  InitializeModuleAndManagers();

  // Run the main "interpreter loop" now.
  MainLoop();

  return 0;
}
```

The KaleidoscopeJIT class is a simple JIT included with this tutorial in
[`code/include/KaleidoscopeJIT.h`](../code/include/KaleidoscopeJIT.h). In later
chapters we will look at how it works and extend it with new features, but for
now we will take it as given. Its API is very simple: `addModule` adds an LLVM
IR module to the JIT, making its functions available for execution (with its
memory managed by a `ResourceTracker`); and `lookup` allows us to look up
pointers to the compiled code.

The KaleidoscopeJIT accepts LLVM IR modules, not MLIR modules, so before
we can add a module to the JIT we need to lower it. This happens in two
steps. First, we use MLIR conversion passes to lower the `func` and
`arith` dialects to the LLVM dialect. The LLVM dialect is still MLIR,
but its operations and types closely represent LLVM IR. We then translate
the resulting MLIR module into an LLVM IR module that can be handed to
the JIT.

This is mostly boilerplate that connects MLIR's lowering and translation
infrastructure to the LLVM JIT. The details are shown here for
completeness, but the same basic process can be reused whenever an MLIR
module is lowered to LLVM IR for execution.

```cpp
static llvm::Expected<llvm::orc::ThreadSafeModule> lowerToLLVM() {
  // Lower the high-level MLIR operations to the LLVM dialect.
  PassManager LoweringPM(TheContext.get());
  LoweringPM.addPass(createConvertFuncToLLVMPass());
  LoweringPM.addPass(createArithToLLVMConversionPass());

  // Clean up any temporary casts introduced by dialect conversion.
  LoweringPM.addPass(createReconcileUnrealizedCastsPass());
  if (failed(LoweringPM.run(*TheModule)))
    return llvm::make_error<llvm::StringError>(
        "could not lower module to the LLVM dialect",
        llvm::inconvertibleErrorCode());

  // Register the translations from MLIR's LLVM dialect to LLVM IR.
  registerBuiltinDialectTranslation(*TheContext);
  registerLLVMDialectTranslation(*TheContext);

  // Translate the lowered MLIR module into an LLVM IR module. The LLVM
  // context is kept with the module because the JIT may compile it later.
  auto LLVMContext = std::make_unique<llvm::LLVMContext>();
  auto LLVMModule = translateModuleToLLVMIR(*TheModule, *LLVMContext);
  if (!LLVMModule)
    return llvm::make_error<llvm::StringError>(
        "could not translate the LLVM dialect to LLVM IR",
        llvm::inconvertibleErrorCode());

  // Match the module's data layout to the target selected by the JIT.
  LLVMModule->setDataLayout(TheJIT->getDataLayout());

  if (DumpLLVMIR) {
    LLVMModule->print(llvm::errs(), nullptr);
    llvm::errs() << '\n';
  }

  // ThreadSafeModule transfers ownership of both objects to the ORC JIT.
  return llvm::orc::ThreadSafeModule(std::move(LLVMModule),
                                     std::move(LLVMContext));
}
```

## The `--dump-llvm-ir` Option

Just as `--dump-mlir` lets us inspect the MLIR produced by the front-end,
`--dump-llvm-ir` lets us inspect the LLVM IR produced by `lowerToLLVM()` before
it is handed to the JIT. The option is defined using LLVM's command-line
support:

```cpp
static llvm::cl::opt<bool> DumpLLVMIR(
    "dump-llvm-ir", llvm::cl::desc("Print translated LLVM IR"),
    llvm::cl::init(false));
```

After translating the module and setting its data layout, `lowerToLLVM()`
checks the option and prints the LLVM IR when requested:

```cpp
if (DumpLLVMIR) {
  LLVMModule->print(llvm::errs(), nullptr);
  llvm::errs() << '\n';
}
```

This is useful for seeing the final representation consumed by the ORC JIT and
for diagnosing problems that occur after MLIR lowering.

## Evaluating Top-Level Expressions

We can now change our code that parses top-level expressions to look like this:

```cpp
static llvm::ExitOnError ExitOnErr;
...
static void HandleTopLevelExpression() {
  // Evaluate a top-level expression into an anonymous function.
  if (auto FnAST = ParseTopLevelExpr()) {
    if (auto FnIR = FnAST->codegen()) {
      if (DumpMLIR) {
        fprintf(stderr, "Read top-level expression:\n");
        FnIR.print(llvm::errs(), OpPrintingFlags().assumeVerified());
        fprintf(stderr, "\n");
      }

      // Create a ResourceTracker to track JIT'd memory allocated to our
      // anonymous expression -- that way we can free it after executing.
      auto RT = TheJIT->getMainJITDylib().createResourceTracker();

      // Lower the MLIR module to LLVM IR and add it to the JIT.
      ExitOnErr(TheJIT->addModule(ExitOnErr(lowerToLLVM()), RT));

      // The module has been handed to the JIT, so open a new module to hold
      // subsequent code.
      InitializeModuleAndManagers();

      // Search the JIT for the __anon_expr symbol.
      auto ExprSymbol = ExitOnErr(TheJIT->lookup("__anon_expr"));

      // Get the symbol's address and cast it to the right type (takes no
      // arguments, returns a double) so we can call it as a native function.
      double (*FP)() = ExprSymbol.getAddress().toPtr<double (*)()>();
      fprintf(stderr, "Evaluated to %f\n", FP());

      // Delete the anonymous expression module from the JIT.
      ExitOnErr(RT->remove());
    }
```

If parsing and codegen succeed, the next step is to convert the MLIR module containing
the top-level expression to LLVM IR and add the resulting LLVM IR module to the JIT. We do
this by calling `lowerToLLVM` and passing its result to `addModule`, which
triggers code generation for all the functions in the module. `addModule` also
accepts a `ResourceTracker` which can be used to remove the module from the JIT
later. Once the module has been added to the JIT it can no longer be
modified, so we also open a new module to hold subsequent code by calling
`InitializeModuleAndManagers()`.

Once we've added the module to the JIT we need to get a pointer to the final
generated code. We do this by calling the JIT's `lookup` method, and passing
the name of the top-level expression function: `__anon_expr`.

Next, we get the in-memory address of the `__anon_expr` function. Recall
that we compile top-level expressions into a self-contained LLVM function that
takes no arguments and returns the computed double. Because the LLVM JIT compiler
matches the native platform ABI, this means that you can just cast the result pointer
to a function pointer of that type and call it directly. This means, there is no
difference between JIT compiled code and native machine code that is statically
linked into your application.

Finally, since we don't support re-evaluation of top-level expressions, we
remove the module from the JIT when we're done to free the associated memory.
Recall, however, that the module we created a few lines earlier (via
`InitializeModuleAndManagers`) is still open and waiting for new code to be
added.

With just these changes, let's see how Kaleidoscope works now!

<!-- code-merge:start -->
```bash
$ build/toy --dump-mlir
```
```kaleidoscope
ready> 4+5;
```
```text
Read top-level expression:
```
```mlir
func.func @__anon_expr() -> f64 {
  %cst = arith.constant 9.000000e+00 : f64
  return %cst : f64
}
```
```text
Evaluated to 9.000000
```
<!-- code-merge:end -->

Well this looks like it is basically working. This demonstrates very
basic functionality, but can we do more?

```mlir
ready> def testfunc(x y) x + y*2;
Read function definition:
func.func @testfunc(%arg0: f64, %arg1: f64) -> f64 {
  %cst = arith.constant 2.000000e+00 : f64
  %0 = arith.mulf %arg1, %cst : f64
  %1 = arith.addf %arg0, %0 : f64
  return %1 : f64
}

ready> testfunc(4, 10);
Evaluated to 24.000000

ready> testfunc(5, 10);
JIT session error: Symbols not found: [ testfunc ]
```

Function definitions and calls also work, but something went very wrong on that
last line. The call looks valid, so what happened? As you may have guessed from
the API a Module is a unit of allocation for the JIT, and testfunc was part
of the same module that contained anonymous expression. When we removed that
module from the JIT to free the memory for the anonymous expression, we deleted
the definition of `testfunc` along with it. Then, when we tried to call
testfunc a second time, the JIT could no longer find it.

The easiest way to fix this is to put the anonymous expression in a separate
module from the rest of the function definitions. The JIT will happily resolve
function calls across module boundaries, as long as each of the functions called
has a prototype, and is added to the JIT before it is called. By putting the
anonymous expression in a different module we can delete it without affecting
the rest of the functions.

In fact, we're going to go a step further and put every function in its own
module. Doing so allows the JIT to keep function definitions while temporary
modules containing top-level expressions are removed:

```mlir
ready> def addOne(x) x + 1;
Read function definition:
func.func @addOne(%arg0: f64) -> f64 {
  %cst = arith.constant 1.000000e+00 : f64
  %0 = arith.addf %arg0, %cst : f64
  return %0 : f64
}

ready> addOne(2);
Evaluated to 3.000000

ready> def addTwo(x) x + 2;
Read function definition:
func.func @addTwo(%arg0: f64) -> f64 {
  %cst = arith.constant 2.000000e+00 : f64
  %0 = arith.addf %arg0, %cst : f64
  return %0 : f64
}

ready> addTwo(2);
Evaluated to 4.000000
```

To allow each function to live in its own module we'll need a way to re-generate previous function declarations into each new module we open:

```cpp
static std::unique_ptr<llvm::orc::KaleidoscopeJIT> TheJIT;
static std::map<std::string, std::unique_ptr<PrototypeAST>> FunctionProtos;

...

func::FuncOp getFunction(const std::string &Name) {
  // First, see if the function has already been added to the current module.
  if (auto Function = TheModule->lookupSymbol<func::FuncOp>(Name))
    return Function;

  // If not, codegen the declaration from an existing prototype.
  auto It = FunctionProtos.find(Name);
  if (It != FunctionProtos.end()) {
    auto Function = It->second->codegen();
    Function.setPrivate();
    return Function;
  }

  return {};
}

...

Value CallExprAST::codegen() {
  // Look up the name in the global module table.
  auto CalleeF = getFunction(Callee);

...

func::FuncOp FunctionAST::codegen() {
  // Save the prototype so declarations can be emitted in later modules.
  auto &P = *Proto;
  FunctionProtos[Proto->getName()] = std::move(Proto);
  auto TheFunction = getFunction(P.getName());
  if (!TheFunction)
    return {};
```

To enable this, we'll start by adding a new global, `FunctionProtos`, that
holds the most recent prototype for each function. We'll also add a convenience
method, `getFunction()`, to replace calls to `TheModule->lookupSymbol<func::FuncOp>()`.
Our convenience method searches `TheModule` for an existing function
declaration, falling back to generating a new declaration from FunctionProtos if
it doesn't find one. In `CallExprAST::codegen()` we just need to replace the
call to `TheModule->lookupSymbol<func::FuncOp>()`. In `FunctionAST::codegen()` we need to
update the FunctionProtos map first, then call `getFunction()`. With this
done, we can always obtain a function declaration in the current module for any
previously declared function.

We also need to update HandleDefinition and HandleExtern:

```cpp
static void HandleDefinition() {
  if (auto FnAST = ParseDefinition()) {
    if (auto FnIR = FnAST->codegen()) {
      if (DumpMLIR) {
        fprintf(stderr, "Read function definition:\n");
        FnIR.print(llvm::errs(), OpPrintingFlags().assumeVerified());
        fprintf(stderr, "\n");
      }

      ExitOnErr(TheJIT->addModule(ExitOnErr(lowerToLLVM())));
      InitializeModuleAndManagers();
    }
  } else {
    // Skip token for error recovery.
    getNextToken();
  }
}

static void HandleExtern() {
  if (auto ProtoAST = ParseExtern()) {
    if (auto FnIR = ProtoAST->codegen()) {
      FnIR.setPrivate();
      if (DumpMLIR) {
        fprintf(stderr, "Read extern:\n");
        FnIR.print(llvm::errs(), OpPrintingFlags().assumeVerified());
        fprintf(stderr, "\n");
      }
      FunctionProtos[ProtoAST->getName()] = std::move(ProtoAST);
    }
  } else {
    // Skip token for error recovery.
    getNextToken();
  }
}
```

In `HandleDefinition`, we add two lines to transfer the newly defined function to the JIT and open a new module. In HandleExtern, we just need to add one line to add the prototype to FunctionProtos.

!!!note
    Duplication of symbols in separate modules is not allowed since LLVM-9. That means you can not redefine function in your Kaleidoscope JIT. The reason is that the newer OrcV2 JIT APIs are trying to stay very close to the static and dynamic linker rules, including rejecting duplicate symbols. Requiring symbol names to be unique allows us to support concurrent compilation for symbols using the (unique) symbol names as keys for tracking.

With these changes made, let's try our REPL again (I removed the dump of the anonymous functions this time, you should get the idea by now :) :

```text
ready> def addOne(x) x + 1;
ready> addOne(2);
Evaluated to 3.000000

ready> def addTwo(x) x + 2;
ready> addTwo(2);
Evaluated to 4.000000
```

It works!

Even with this simple code, we get some surprisingly powerful capabilities -
check this out:

<!-- code-merge:start -->
```bash
$ build/toy
```
```kaleidoscope
ready> extern sin(x);
ready> extern cos(x);
ready> sin(1.0);
```
```text
Evaluated to 0.841471
```
```kaleidoscope
ready> def pythagoreanIdentity(x) sin(x)*sin(x) + cos(x)*cos(x);
ready> pythagoreanIdentity(4.0);
```
```text
Evaluated to 1.000000
```
<!-- code-merge:end -->

Whoa, how does the JIT know about sin and cos? The answer is surprisingly
simple: The KaleidoscopeJIT has a straightforward symbol resolution rule that
it uses to find symbols that aren't available in any given module: First it
searches the definitions that have already been added to the JIT. If no
definition is found inside the JIT, it falls back to searching the
Kaleidoscope process itself. Since "`sin`" is available in the host process,
the JIT resolves the call to the libm version of `sin`. The "`sin(1.0)`"
expression above therefore executes that function at runtime.

In the future we'll see how tweaking this symbol resolution rule can be used to
enable all sorts of useful features, from security (restricting the set of
symbols available to JIT'd code), to dynamic code generation based on symbol
names, and even lazy compilation.

One immediate benefit of the symbol resolution rule is that we can now extend
the language by writing arbitrary C++ code to implement operations. For example,
if we add:

```cpp
#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

/// putchard - putchar that takes a double and returns 0.
extern "C" DLLEXPORT double putchard(double X) {
  fputc((char)X, stderr);
  return 0;
}
```

Note, that for Windows we need to actually export the functions because
the dynamic symbol loader will use `GetProcAddress` to find the symbols.

Now we can produce simple output to the console by using things like:
"`extern putchard(x); putchard(120);`", which prints a lowercase 'x'
on the console (120 is the ASCII code for 'x'). Similar code could be
used to implement file I/O, console input, and many other capabilities
in Kaleidoscope.

This completes the JIT and optimizer chapter of the Kaleidoscope
tutorial. At this point, we can compile a non-Turing-complete
programming language, optimize and JIT compile it in a user-driven way.
Next up we'll look into [extending the language with control flow
constructs](chapter-05.md), tackling some interesting MLIR issues
along the way.

## Full Code Listing

We use the following `CMakeLists.txt` to build the example:

```cmake(../code/chapter-03/CmakeLists.txt)
```

The `ENABLE_EXPORTS` property makes symbols in the executable available
for runtime lookup by the JIT. CMake supplies the appropriate linker
option for the platform, including `-rdynamic` where it is required on
Linux.

Once you have built MLIR, point `MLIR_DIR` at the directory containing
`MLIRConfig.cmake`, then build and run the example:

```bash
cmake -S . -B build \
  -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir
cmake --build build
./build/toy
```

Here is the code:

```cpp(../code/chapter-04/toy.cpp)
```

[Next: Extending the language: control flow](chapter-05.md)
