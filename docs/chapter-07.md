# 7. Kaleidoscope: Extending the Language: Mutable Variables

## Chapter 7 Introduction

Welcome to Chapter 7 of the "[Implementing a language with MLIR](chapter-00.md)" tutorial. In chapters 1 through 6, we've built a very respectable, albeit simple, [functional programming language](http://en.wikipedia.org/wiki/Functional_programming). In our journey, we learned some parsing techniques, how to build and represent an AST, how to build MLIR, lower it to LLVM IR, optimize it, and JIT compile it.

While Kaleidoscope is interesting as a functional language, the fact that it is functional makes it "too easy" to generate SSA-based IR for it. In particular, a functional language makes it very easy to build values directly in [SSA form](http://en.wikipedia.org/wiki/Static_single_assignment_form). MLIR also represents values in SSA form, so this is a very nice property and it is often unclear to newcomers how to generate code for an imperative language with mutable variables.

The short (and happy) summary of this chapter is that there is no need for your front-end to build SSA form: MLIR provides reusable and well-tested support for this, though the way it works is a bit unexpected for some.

## Why is this a hard problem?

To understand why mutable variables cause complexities in SSA construction, consider this extremely simple C example:

```c
int G, H;
int test(_Bool Condition) {
  int X;
  if (Condition)
    X = G;
  else
    X = H;
  return X;
}
```

In this case, we have the variable "X", whose value depends on the path executed in the program. Because there are two different possible values for X before the return instruction, a PHI node is inserted to merge the two values. The LLVM IR that we want for this example looks like this:

```llvm
@G = weak global i32 0   ; type of @G is i32*
@H = weak global i32 0   ; type of @H is i32*

define i32 @test(i1 %Condition) {
entry:
  br i1 %Condition, label %cond_true, label %cond_false

cond_true:
  %X.0 = load i32, i32* @G
  br label %cond_next

cond_false:
  %X.1 = load i32, i32* @H
  br label %cond_next

cond_next:
  %X.2 = phi i32 [ %X.1, %cond_false ], [ %X.0, %cond_true ]
  ret i32 %X.2
}
```

In this example, the loads from the G and H global variables are explicit in the LLVM IR, and they live in the then/else branches of the if statement (cond_true/cond_false). In order to merge the incoming values, the X.2 phi node in the cond_next block selects the right value to use based on where control flow is coming from: if control flow comes from the cond_false block, X.2 gets the value of X.1. Alternatively, if control flow comes from cond_true, it gets the value of X.0. The intent of this chapter is not to explain the details of SSA form. For more information, see one of the many [online references](http://en.wikipedia.org/wiki/Static_single_assignment_form).

The question for this chapter is: who places the MLIR block arguments when lowering assignments to mutable variables? As we saw in Chapter 5, we can solve this problem entirely in MLIR. The issue here is that MLIR represents values in SSA form: there is no "non-SSA" mode for values. However, SSA construction requires non-trivial algorithms and data structures, so it is inconvenient and wasteful for every frontend to have to reproduce this logic.

## Memory in MLIR

The "trick" here is that while MLIR does require all values to be in SSA form, it does not require the contents of memory to be in SSA form. In the example above, note that the loads from G and H are direct accesses to G and H: the memory objects are not renamed or versioned. This differs from some other compiler systems, which do try to version memory objects. In MLIR, instead of encoding dataflow analysis of memory into the IR, it is handled with [analyses and passes](https://mlir.llvm.org/docs/PassManagement/#analysis-management) that are computed when needed.

With this in mind, the high-level idea is that we want to make a stack variable (which lives in memory, because it is on the stack) for each mutable object in a function. To take advantage of this trick, we need to talk about how MLIR represents stack variables.

In MLIR, all memory accesses are explicit with `memref.load` and `memref.store` operations, and there is no need for an "address-of" operator at this level. Notice how the type of `%x` in the example below is actually `memref<f64>` even though the variable holds an `f64`. What this means is that `%x` identifies *space* for an `f64`, and its name refers to that storage rather than to the value currently stored there. Stack variables are declared with the [memref.alloca](https://mlir.llvm.org/docs/Dialects/MemRef/#memrefalloca-memrefallocaop) operation:

```mlir
%x = memref.alloca() : memref<f64>
memref.store %initial, %x[] : memref<f64>
%value = memref.load %x[] : memref<f64>
```

This code shows an example of how you can declare and manipulate a stack variable in MLIR.  Stack memory allocated with `memref.alloca` is fully general: you can pass the memref to functions, create aliases or views of it, and use it with other memory operations. In our example above, we could rewrite the example to use the `memref.alloca` technique to avoid creating block arguments in the frontend:

```mlir
// example.mlir
module {
  func.func @test(%condition: i1, %g: f64, %h: f64) -> f64 {
    %x = memref.alloca() : memref<f64>
    scf.if %condition {
      memref.store %g, %x[] : memref<f64>
    } else {
      memref.store %h, %x[] : memref<f64>
    }
    %value = memref.load %x[] : memref<f64>
    return %value : f64
  }
}
```

If we lower this directly to LLVM IR without running `mem2reg`, the stack slot and its accesses remain. The `insertvalue` and `extractvalue` instructions construct and access the descriptor used while lowering the memref, but the important operations here are the `alloca`, the store in each branch, and the load after the branches merge:

```llvm
define double @test(i1 %0, double %1, double %2) {
  %4 = alloca double, i64 1, align 8
  %5 = insertvalue { ptr, ptr, i64 } poison, ptr %4, 0
  %6 = insertvalue { ptr, ptr, i64 } %5, ptr %4, 1
  %7 = insertvalue { ptr, ptr, i64 } %6, i64 0, 2
  br i1 %0, label %8, label %10

8:
  %9 = extractvalue { ptr, ptr, i64 } %7, 1
  store double %1, ptr %9, align 8
  br label %12

10:
  %11 = extractvalue { ptr, ptr, i64 } %7, 1
  store double %2, ptr %11, align 8
  br label %12

12:
  %13 = extractvalue { ptr, ptr, i64 } %7, 1
  %14 = load double, ptr %13, align 8
  ret double %14
}
```

With this, we have discovered a way to handle arbitrary mutable variables without the need to create block arguments at all:

1. Each mutable variable becomes a stack allocation.
2. Each read of the variable becomes a load from the stack.
3. Each update of the variable becomes a store to the stack.
4. Passing a variable's storage uses the memref directly.

While this solution has solved our immediate problem, it introduced another one: we have now apparently introduced a lot of stack traffic for very simple and common operations, a major performance problem. Fortunately for us, MLIR has a `mem2reg` pass that handles this case, promoting suitable memory slots into SSA values and inserting block arguments as appropriate.

The `mem2reg` pass operates on control-flow blocks, so we first lower `scf.if` to the `cf` dialect. If you run this example through the passes, you'll get:

```bash
mlir-opt example.mlir --convert-scf-to-cf --mem2reg
```

```mlir
module {
  func.func @test(%arg0: i1, %arg1: f64, %arg2: f64) -> f64 {
    cf.cond_br %arg0, ^bb1, ^bb2
  ^bb1:
    cf.br ^bb3(%arg1 : f64)
  ^bb2:
    cf.br ^bb3(%arg2 : f64)
  ^bb3(%0: f64):
    return %0 : f64
  }
}
```

The `mem2reg` pass implements the standard "iterated dominance frontier" algorithm for constructing SSA form. When multiple stored values can reach a load, it inserts a block argument at the merge point and passes the appropriate value from each predecessor. The `mem2reg` optimization pass is the answer to dealing with mutable variables, and we highly recommend that you depend on it. Note that MLIR's `mem2reg` only works on variables in certain circumstances:

1. `mem2reg` is allocation-driven: it looks for operations that expose promotable memory slots. For this tutorial, those are `memref.alloca` operations. It does not promote global variables or arbitrary heap allocations.
2. The allocation must dominate all of its uses. Creating our stack slots in the function's entry block makes them available throughout the function and makes this condition easy to satisfy.
3. Every use of the memory slot must be understood by the promotion interfaces. Direct `memref.load` and `memref.store` operations work. Passing the memref to an arbitrary function or using it through an unsupported alias prevents promotion.
4. A `memref.alloca` containing one element can be promoted directly to a scalar SSA value. This is why Kaleidoscope uses the zero-dimensional `memref<f64>` type for each mutable variable. MLIR can handle some more complicated memrefs too, but those cases are not needed here.

All of these properties are easy to satisfy for most imperative languages, and we'll illustrate it below with Kaleidoscope. The final question you may be asking is: should I bother with this nonsense for my frontend? Wouldn't it be better if I just did SSA construction directly, avoiding use of the `mem2reg` optimization pass? In short, we strongly recommend that you use this technique for building SSA form, unless there is an extremely good reason not to. Using this technique is:

- **Proven and well tested:** MLIR provides the shared SSA-construction algorithm and promotion interfaces, so each frontend does not have to implement and maintain its own version.
- **Extremely fast:** `mem2reg` forwards stored values directly to their uses and only adds block arguments at the merge points where they are needed.
- **Needed for debug info generation:** keeping a variable in memory gives it a concrete address to which debug information can be attached. In Chapter 9, we retain that storage in unoptimized builds so the debugger can show our variables.

If nothing else, this makes it much easier to get your frontend up and running, and is very simple to implement. Let's extend Kaleidoscope with mutable variables now!

## Mutable Variables in Kaleidoscope

Now that we know the sort of problem we want to tackle, let's see what this looks like in the context of our little Kaleidoscope language. We're going to add two features:

1. The ability to mutate variables with the '=' operator.
2. The ability to define new variables.

While the first item is really what this is about, we only have variables for incoming arguments as well as for induction variables, and redefining those only goes so far :). Also, the ability to define new variables is a useful thing regardless of whether you will be mutating them. Here's a motivating example that shows how we could use these:

```kaleidoscope
# Define ':' for sequencing: as a low-precedence operator that ignores operands
# and just returns the RHS.
def binary : 1 (x y) y;

# Recursive fib, we could do this before.
def fib(x)
  if (x < 3) then
    1
  else
    fib(x-1)+fib(x-2);

# Iterative fib.
def fibi(x)
  var a = 1, b = 1, c in
  (for i = 3, i < x + 1 in
     c = a + b :
     a = b :
     b = c) :
  b;

# Call it.
fibi(10);
```

In order to mutate variables, we will change existing variables to use MLIR memory operations. Once that works, we will add assignment and then extend Kaleidoscope with local variable definitions.

## Adjusting Existing Variables for Mutation

The symbol table in Kaleidoscope is managed at code generation time by the `NamedValues` map. This map currently keeps track of the MLIR `Value` that holds the `f64` value for the named variable. In order to support mutation, we need to change this slightly, so that `NamedValues` holds the *memory location* of the variable in question. Note that this change is a refactoring: it changes the structure of the code, but does not (by itself) change the behavior of the compiler. All of these changes are isolated in the Kaleidoscope code generator.

At this point in Kaleidoscope's development, it only supports variables for two things: incoming arguments to functions and the induction variable of `for` loops. For consistency, we'll allow mutation of these variables in addition to other user-defined variables. This means that these will both need memory locations.

To start our transformation of Kaleidoscope, we'll change the meaning of the values stored in the `NamedValues` map. Instead of mapping each name directly to its current `f64` SSA value, it will map the name to a zero-dimensional memref containing the variable. Both are represented by MLIR's `Value` C++ type, so the declaration itself does not change:

```cpp
static std::map<std::string, Value> NamedValues;
```

We first find the function surrounding the builder's current insertion point, then use a helper to create storage at the beginning of that function:

```cpp
static func::FuncOp getCurrentFunction() {
  Operation *Parent = TheBuilder->getInsertionBlock()->getParentOp();
  if (auto Function = dyn_cast<func::FuncOp>(Parent))
    return Function;
  return Parent->getParentOfType<func::FuncOp>();
}

/// CreateEntryBlockStorage - Create mutable storage in the function entry block.
static Value CreateEntryBlockStorage() {
  func::FuncOp Function = getCurrentFunction();
  OpBuilder::InsertionGuard Guard(*TheBuilder);
  TheBuilder->setInsertionPointToStart(&Function.front());
  auto VariableType = MemRefType::get({}, TheBuilder->getF64Type());
  return TheBuilder->create<memref::AllocaOp>(getLocation(), VariableType);
}
```

`InsertionGuard` restores the builder's previous insertion point when the helper returns. Placing the allocation at the start of the entry block makes the storage available to every region in the function.

A variable reference now looks up its storage and loads the current value:

```cpp
Value VariableExprAST::codegen() {
  auto It = NamedValues.find(Name);
  if (It == NamedValues.end())
    return LogErrorV("Unknown variable name");

  return TheBuilder->create<memref::LoadOp>(getLocation(), It->second,
                                             ValueRange{});
}
```

Loop induction variables use the same representation. We allocate a slot, store the initial value, and make the slot visible through `NamedValues`:

```cpp
Value Variable = CreateEntryBlockStorage();
TheBuilder->create<memref::StoreOp>(getLocation(), StartVal, Variable,
                                    ValueRange{});

auto OldValue = NamedValues.find(VarName);
bool HadOldValue = OldValue != NamedValues.end();
Value SavedValue = HadOldValue ? OldValue->second : Value();
NamedValues[VarName] = Variable;
```

After the loop body and step have run, we reload the induction variable in case either expression changed it, compute the next value, and store it:

```cpp
Value Current =
    Builder.create<memref::LoadOp>(Loc, Variable, ValueRange{});
Value NextVar = Builder.create<arith::AddFOp>(Loc, Current, StepVal);
Builder.create<memref::StoreOp>(Loc, NextVar, Variable, ValueRange{});
```

Function arguments also become mutable storage when a function body is created:

```cpp
NamedValues.clear();
unsigned Index = 0;
for (BlockArgument Argument : TheFunction.getArguments()) {
  Value Storage = CreateEntryBlockStorage();
  TheBuilder->create<memref::StoreOp>(getLocation(), Argument, Storage,
                                      ValueRange{});
  NamedValues[P.getArgs()[Index++]] = Storage;
}
```

Finally, the lowering pipeline converts structured control flow to blocks and runs MLIR's `mem2reg` pass before lowering the remaining operations to the LLVM dialect:

```cpp
PassManager LoweringPM(TheContext.get());
LoweringPM.addPass(createSCFToControlFlowPass());
LoweringPM.addPass(createMem2Reg());
LoweringPM.addPass(createConvertFuncToLLVMPass());
LoweringPM.addPass(createArithToLLVMConversionPass());
LoweringPM.addPass(createFinalizeMemRefToLLVMConversionPass());
LoweringPM.addPass(createConvertControlFlowToLLVMPass());
LoweringPM.addPass(createReconcileUnrealizedCastsPass());
```

Now all variable references consistently go through mutable storage, while the promotion pass can recover SSA values before translation to LLVM IR.

## New Assignment Operator

With our current framework, adding an assignment operator is simple. We parse it like any other binary operator, but handle it internally instead of allowing the user to define it. First we give it a low precedence:

```cpp
int main() {
  // Install standard binary operators.
  // 1 is lowest precedence.
  BinopPrecedence['='] = 2;
  BinopPrecedence['<'] = 10;
  BinopPrecedence['+'] = 20;
  BinopPrecedence['-'] = 20;
```

Assignment cannot follow the usual “emit the LHS, emit the RHS, perform the operation” pattern because evaluating the LHS would load its value. Instead, we ask whether the left-hand expression names a variable, generate the new value, and store it into that variable's memref:

```cpp
Value BinaryExprAST::codegen() {
  // Assignment stores into the variable's mutable memref slot.
  if (Op == '=') {
    const std::string *Name = LHS->getVariableName();
    if (!Name)
      return LogErrorV("destination of '=' must be a variable");

    Value AssignedValue = RHS->codegen();
    if (!AssignedValue)
      return {};

    auto It = NamedValues.find(*Name);
    if (It == NamedValues.end())
      return LogErrorV("Unknown variable name");

    TheBuilder->create<memref::StoreOp>(getLocation(), AssignedValue,
                                        It->second, ValueRange{});
    return AssignedValue;
  }
```

Returning the assigned value permits chained assignments such as `x = (y = z)`. We can now mutate loop variables and function arguments:

```kaleidoscope
# Function to print a double.
extern printd(x);

# Define ':' for sequencing.
def binary : 1 (x y) y;

def test(x)
  printd(x) :
  x = 4 :
  printd(x);

test(123);
```

When run, this prints `123` and then `4`. We can mutate existing variables; next we will add declarations for new local variables.

## User-defined Local Variables

Adding var/in is just like any other extension we made to Kaleidoscope: we extend the lexer, the parser, the AST and the code generator. The first step for adding our new 'var/in' construct is to extend the lexer. As before, this is pretty trivial, the code looks like this:

```cpp
enum Token {
  ...
  // var definition
  tok_var = -13
...
}
...
static int gettok() {
...
    if (IdentifierStr == "in")
      return tok_in;
    if (IdentifierStr == "binary")
      return tok_binary;
    if (IdentifierStr == "unary")
      return tok_unary;
    if (IdentifierStr == "var")
      return tok_var;
    return tok_identifier;
...
```

The next step is to define the AST node that we will construct. For var/in, it looks like this:

```cpp
/// VarExprAST - Expression class for var/in
class VarExprAST : public ExprAST {
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
  std::unique_ptr<ExprAST> Body;

public:
  VarExprAST(
      std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
      std::unique_ptr<ExprAST> Body)
      : VarNames(std::move(VarNames)), Body(std::move(Body)) {}

  Value codegen() override;
};
```

var/in allows a list of names to be defined all at once, and each name can optionally have an initializer value. As such, we capture this information in the VarNames vector. Also, var/in has a body, this body is allowed to access the variables defined by the var/in.

With this in place, we can define the parser pieces. The first thing we do is add it as a primary expression:

```cpp
/// primary
///   ::= identifierexpr
///   ::= numberexpr
///   ::= parenexpr
///   ::= ifexpr
///   ::= forexpr
///   ::= varexpr
static std::unique_ptr<ExprAST> ParsePrimary() {
  switch (CurTok) {
  default:
    return LogError("unknown token when expecting an expression");
  case tok_identifier:
    return ParseIdentifierExpr();
  case tok_number:
    return ParseNumberExpr();
  case '(':
    return ParseParenExpr();
  case tok_if:
    return ParseIfExpr();
  case tok_for:
    return ParseForExpr();
  case tok_var:
    return ParseVarExpr();
  }
}
```

Next we define ParseVarExpr:

```cpp
/// varexpr ::= 'var' identifier ('=' expression)?
//                    (',' identifier ('=' expression)?)* 'in' expression
static std::unique_ptr<ExprAST> ParseVarExpr() {
  getNextToken(); // eat the var.

  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
  if (CurTok != tok_identifier)
    return LogError("expected identifier after var");
```

The first part of this code parses the list of identifier/expr pairs into the local `VarNames` vector.

```cpp
while (true) {
  std::string Name = IdentifierStr;
  getNextToken(); // eat identifier.

  std::unique_ptr<ExprAST> Init;
  if (CurTok == '=') {
    getNextToken(); // eat '='.
    Init = ParseExpression();
    if (!Init)
      return nullptr;
  }

  VarNames.emplace_back(Name, std::move(Init));

  if (CurTok != ',')
    break;
  getNextToken(); // eat ','.
  if (CurTok != tok_identifier)
    return LogError("expected identifier list after var");
}
```

Once all the variables are parsed, we then parse the body and create the AST node:

```cpp
  if (CurTok != tok_in)
    return LogError("expected 'in' keyword after 'var'");
  getNextToken(); // eat 'in'.

  auto Body = ParseExpression();
  if (!Body)
    return nullptr;

  return std::make_unique<VarExprAST>(std::move(VarNames), std::move(Body));
}
```

Now that we can parse and represent the code, we generate mutable storage for each local variable:

```cpp
Value VarExprAST::codegen() {
  std::vector<std::pair<std::string, std::optional<Value>>> OldBindings;

  auto RestoreBindings = [&]() {
    for (auto It = OldBindings.rbegin(); It != OldBindings.rend(); ++It) {
      if (It->second)
        NamedValues[It->first] = *It->second;
      else
        NamedValues.erase(It->first);
    }
  };

  for (auto &Variable : VarNames) {
    const std::string &Name = Variable.first;

    // Generate the initializer before introducing the new binding.
    Value InitialValue;
    if (Variable.second)
      InitialValue = Variable.second->codegen();
    else
      InitialValue = TheBuilder->create<arith::ConstantOp>(
          getLocation(), TheBuilder->getF64FloatAttr(0.0));
    if (!InitialValue) {
      RestoreBindings();
      return {};
    }

    Value Storage = CreateEntryBlockStorage();
    TheBuilder->create<memref::StoreOp>(getLocation(), InitialValue, Storage,
                                        ValueRange{});

    auto Old = NamedValues.find(Name);
    OldBindings.emplace_back(
        Name, Old == NamedValues.end() ? std::optional<Value>()
                                      : std::optional<Value>(Old->second));
    NamedValues[Name] = Storage;
  }

  Value BodyValue = Body->codegen();
  RestoreBindings();
  return BodyValue;
}
```

The initializer is generated before the new name is installed, so an inner declaration such as `var a = a in ...` reads the outer `a`. We remember every previous binding and restore them in reverse order when the body is finished.

With this, the iterative Fibonacci example from the introduction compiles and runs. The frontend expresses mutation with straightforward memory operations; MLIR's `mem2reg` pass promotes those operations into SSA values and introduces block arguments where different control-flow paths meet.

## Full Code Listing

Here is the complete code listing for our running example, enhanced with mutable variables and var/in support. Here is the CMake configuration:

```cmake(../code/chapter-07/CMakeLists.txt)
```

To build this example, use:

```bash
cmake -S . -B build \
  -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir
cmake --build build
./build/toy
```

Here is the code:

```cpp(../code/chapter-07/toy.cpp)
```

[Next: Compiling to Object Code](chapter-08.md)
