# 5. Kaleidoscope: Extending the Language: Control Flow

## Chapter 5 Introduction

Welcome to Chapter 5 of the "[Implementing a language with MLIR](chapter-00.md)" tutorial. Parts 1-4 described the implementation of
the simple Kaleidoscope language and included support for generating
MLIR, followed by optimizations and a JIT compiler. Unfortunately, as
presented, Kaleidoscope is mostly useless: it has no control flow other
than call and return. This means that you can't have conditional
branches in the code, significantly limiting its power. In this episode
of "build that compiler", we'll extend Kaleidoscope to have an
if/then/else expression plus a simple 'for' loop.

## If/Then/Else

Extending Kaleidoscope to support if/then/else is quite straightforward.
It basically requires adding support for this "new" concept to the
lexer, parser, AST, and using an additional MLIR dialect. This example is nice, because
it shows how easy it is to "grow" a language over time, incrementally
extending it as new ideas are discovered.

Before we get going on "how" we add this extension, let's talk about
"what" we want. The basic idea is that we want to be able to write this
sort of thing:

```kaleidoscope
def fib(x)
  if x < 3 then
    1
  else
    fib(x-1)+fib(x-2);
```

In Kaleidoscope, every construct is an expression: there are no
statements. As such, the if/then/else expression needs to return a value
like any other. Since we're using a mostly functional form, we'll have
it evaluate its conditional, then return the 'then' or 'else' value
based on how the condition was resolved. This is very similar to the C
"?:" expression.

The semantics of the if/then/else expression is that it evaluates the
condition to a boolean equality value: 0.0 is considered to be false and
everything else is considered to be true. If the condition is true, the
first subexpression is evaluated and returned, if the condition is
false, the second subexpression is evaluated and returned. Since
Kaleidoscope allows side-effects, this behavior is important to nail
down.

Now that we know what we "want", let's break this down into its
constituent pieces.

### Lexer Extensions for If/Then/Else

The lexer extensions are straightforward. First we add new enum values
for the relevant tokens:

```cpp
// control
tok_if = -6,
tok_then = -7,
tok_else = -8,
```

Once we have that, we recognize the new keywords in the lexer. This is
pretty simple stuff:

```cpp
...
if (IdentifierStr == "def")
  return tok_def;
if (IdentifierStr == "extern")
  return tok_extern;
if (IdentifierStr == "if")
  return tok_if;
if (IdentifierStr == "then")
  return tok_then;
if (IdentifierStr == "else")
  return tok_else;
return tok_identifier;
```

### AST Extensions for If/Then/Else

To represent the new expression we add a new AST node for it:

```cpp
/// IfExprAST - Expression class for if/then/else.
class IfExprAST : public ExprAST {
  std::unique_ptr<ExprAST> Cond, Then, Else;

public:
  IfExprAST(std::unique_ptr<ExprAST> Cond, std::unique_ptr<ExprAST> Then,
            std::unique_ptr<ExprAST> Else)
    : Cond(std::move(Cond)), Then(std::move(Then)), Else(std::move(Else)) {}

  Value codegen() override;
};
```

The AST node just has pointers to the various subexpressions.

### Parser Extensions for If/Then/Else

Now that we have the relevant tokens coming from the lexer and we have
the AST node to build, our parsing logic is relatively straightforward.
First we define a new parsing function:

```cpp
/// ifexpr ::= 'if' expression 'then' expression 'else' expression
static std::unique_ptr<ExprAST> ParseIfExpr() {
  getNextToken();  // eat the if.

  // condition.
  auto Cond = ParseExpression();
  if (!Cond)
    return nullptr;

  if (CurTok != tok_then)
    return LogError("expected then");
  getNextToken();  // eat the then

  auto Then = ParseExpression();
  if (!Then)
    return nullptr;

  if (CurTok != tok_else)
    return LogError("expected else");

  getNextToken();

  auto Else = ParseExpression();
  if (!Else)
    return nullptr;

  return std::make_unique<IfExprAST>(std::move(Cond), std::move(Then),
                                      std::move(Else));
}
```

Next we hook it up as a primary expression:

```cpp
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
  }
}
```

### MLIR for If/Then/Else

Now that we have it parsing and building the AST, the final piece is
adding MLIR code generation support. This is the most interesting part
of the if/then/else example, because this is where it starts to
introduce new concepts. All of the code above has been thoroughly
described in previous chapters.

To motivate the code we want to produce, let's take a look at a simple
example. Consider:

```kaleidoscope
extern foo();
extern bar();
def baz(x) if x then foo() else bar();
```

The MLIR you’ll (soon) get from Kaleidoscope looks like this:

```mlir
ready> extern foo();
Read extern:
func.func private @foo() -> f64
ready> extern bar();
Read extern:
func.func private @bar() -> f64
ready> def baz(x) if x then foo() else bar();
Read function definition:
func.func @baz(%arg0: f64) -> f64 {
  %cst = arith.constant 0.000000e+00 : f64
  %0 = arith.cmpf one, %arg0, %cst : f64
  %1 = scf.if %0 -> (f64) {
    %2 = func.call @foo() : () -> f64
    scf.yield %2 : f64
  } else {
    %2 = func.call @bar() : () -> f64
    scf.yield %2 : f64
  }
  return %1 : f64
}
```

So far, we've been looking only at the MLIR dump for our code generation. We will now descend into the LLVM IR to understand a few useful concepts such as the `Phi node`. It's not something you'll deal with directly right now, but it is something that, as a compiler engineer, you need to be aware of. SO here's an LLVM IR dum of our code. 


```llvm
declare double @foo()
declare double @bar()

define double @baz(double %x) {
entry:
  %ifcond = fcmp one double %x, 0.000000e+00
  ;; Compare x with 0.0, producing an i1 condition.

  br i1 %ifcond, label %then, label %else
  ;; Branch to %then if the condition is true, or %else if it is false.

then:
  %thenvalue = call double @foo()
  br label %ifcont

else:
  %elsevalue = call double @bar()
  br label %ifcont

ifcont:
  %iftmp = phi double [ %elsevalue, %else ], [ %thenvalue, %then ]
  br label %return

return:
  ret double %iftmp
}
```

!!!note
    The LLVM IR dump normally uses numbered names such as `%0`, `%1`, and `%2`. In the listing above, we've replaced those numbers with descriptive names to make the control flow easier to follow. We've also added comments that won't appear in the actual output.

The generated code is fairly simple: the entry block evaluates the conditional expression ("x" in our case here) and compares the result to 0.0 with the "`fcmp one`" instruction ('one' is "Ordered and Not Equal"). Based on the result of this expression, the code jumps to either the "then" or "else" blocks, which contain the expressions for the true/false cases. 

Once the then/else blocks are finished executing, they both branch back to the 'ifcont' block to execute the code that happens after the if/then/else. In this case the only thing left to do is to return to the caller of the function. The question then becomes: how does the code know which expression to return?

The answer to this question involves an important SSA operation: the
[Phi operation](http://en.wikipedia.org/wiki/Static_single_assignment_form).
If you're not familiar with SSA, [the wikipedia
article](http://en.wikipedia.org/wiki/Static_single_assignment_form)
is a good introduction and there are various other introductions to it
available on your favorite search engine. The short version is that
"execution" of the Phi operation requires "remembering" which block
control came from. The Phi operation takes on the value corresponding to
the input control block. In this case, if control comes in from the
"then" block, it gets the value of "calltmp". If control comes from the
"else" block, it gets the value of "calltmp1".

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="images/t-llvm-gray.svg">
  <img src="images/t-llvm.svg" alt="LLVM control-flow graph">
</picture>

At this point, you are probably starting to think "Oh no! This means my
simple and elegant front-end will have to start generating SSA form in
order to use LLVM!". Fortunately, this is not the case, and we strongly
advise *not* implementing an SSA construction algorithm in your
front-end unless there is an amazingly good reason to do so. In
practice, there are two sorts of values that float around in code
written for your average imperative programming language that might need
Phi nodes:

1. Code that involves user variables: `x = 1; x = x + 1;`
2. Values that are implicit in the structure of your AST, such as the
   Phi node in this case.

In [Chapter 7](chapter-07.md) of this tutorial ("mutable variables"),
we'll talk about #1 in depth. For now, just believe me that you don't
need SSA construction to handle this case. For #2, you have the choice
of using the techniques that we will describe for #1, or you can insert
Phi nodes directly, if convenient. Had we been using lower level LLVM instead of MLIR, it is really
easy to generate the Phi node, and we would choose to do it directly.

However, with MLIR, we have to do NONE of this. We use the [`scf`]() dialect and insert if/else operations and it does everything for us!

Okay, enough of the motivation and overview, let's generate code!

### Code Generation for If/Then/Else

In order to generate code for this, we implement the `codegen` method
for `IfExprAST`.

The first part emits the condition:

```cpp
Value IfExprAST::codegen() {
  Value CondV = Cond->codegen();
  if (!CondV)
    return {};

  // Convert the condition to a boolean by comparing it with 0.0.
  Value Zero = TheBuilder->create<arith::ConstantOp>(
      getLocation(), TheBuilder->getF64FloatAttr(0.0));
  CondV = TheBuilder->create<arith::CmpFOp>(
      getLocation(), arith::CmpFPredicate::ONE, CondV, Zero);
```

This code is straightforward and similar to what we saw before. We emit
the expression for the condition, then compare that value to zero to get
an `i1` truth value.

With the condition emitted, we can create an
[`scf.if`](https://mlir.llvm.org/docs/Dialects/SCFDialect/#scfif-scfifop)
operation. The `scf` dialect represents structured control flow, allowing
us to describe the `if` expression directly instead of constructing its
basic blocks ourselves.

The first callback passed to `scf::IfOp` builds the `then` region:

```cpp
  bool CodegenFailed = false;
  auto IfOp = TheBuilder->create<scf::IfOp>(
      getLocation(), CondV,
      [&](OpBuilder &Builder, Location Loc) {
        Value ThenV = Then->codegen();
        if (!ThenV) {
          CodegenFailed = true;
          ThenV = Builder.create<arith::ConstantOp>(
              Loc, Builder.getF64FloatAttr(0.0));
        }
        Builder.create<scf::YieldOp>(Loc, ThenV);
      },
```

We recursively generate the value of the `then` expression and finish
the region with
[`scf.yield`](https://mlir.llvm.org/docs/Dialects/SCFDialect/#scfyield-scfyieldop).
The yielded value becomes the result of the `scf.if` operation when its
condition is true.

Every region of an `scf.if` that produces a result must end with an
`scf.yield` providing that result. If code generation fails, we record
the failure and emit a temporary value so that the region remains
structurally complete.

The second callback builds the `else` region in the same way:

```cpp
      [&](OpBuilder &Builder, Location Loc) {
        Value ElseV = Else->codegen();
        if (!ElseV) {
          CodegenFailed = true;
          ElseV = Builder.create<arith::ConstantOp>(
              Loc, Builder.getF64FloatAttr(0.0));
        }
        Builder.create<scf::YieldOp>(Loc, ElseV);
      });
```

The `then` and `else` regions must yield values of the same type. In
Kaleidoscope, both values are doubles, so the `scf.if` operation itself
produces a single `f64` result.

Finally, we check whether either region failed and return the result of
the `scf.if` operation:

```cpp
  if (CodegenFailed)
    return {};
  return IfOp.getResult(0);
}
```

This result is the value computed by the complete if/then/else
expression. In our example, it is either the value returned by `foo()`
or the value returned by `bar()`.

Overall, we now have the ability to execute conditional code in
Kaleidoscope. With this extension, Kaleidoscope is a fairly complete
language that can calculate a wide variety of numeric functions. Next up
we'll add another useful expression that is familiar from non-functional
languages...

## 'for' Loop Expression

Now that we know how to add basic control flow constructs to the
language, we have the tools to add more powerful things. Let's add
something more aggressive, a 'for' expression:

```kaleidoscope
extern putchard(char);
def printstar(n)
  for i = 1, i < n, 1.0 in
    putchard(42);  # ascii 42 = '*'

# print 100 '*' characters
printstar(100);
```

This expression defines a new variable ("i" in this case) which iterates
from a starting value, while the condition ("i < n" in this case) is
true, incrementing by an optional step value ("1.0" in this case). If
the step value is omitted, it defaults to 1.0. While the loop is true,
it executes its body expression. Because we don't have anything better
to return, we'll just define the loop as always returning 0.0. In the
future when we have mutable variables, it will get more useful.

As before, let's talk about the changes that we need to Kaleidoscope to
support this.

### Lexer Extensions for the 'for' Loop

The lexer extensions are the same sort of thing as for if/then/else:

```cpp
... in enum Token ...
// control
tok_if = -6, tok_then = -7, tok_else = -8,
tok_for = -9, tok_in = -10

... in gettok ...
if (IdentifierStr == "def")
  return tok_def;
if (IdentifierStr == "extern")
  return tok_extern;
if (IdentifierStr == "if")
  return tok_if;
if (IdentifierStr == "then")
  return tok_then;
if (IdentifierStr == "else")
  return tok_else;
if (IdentifierStr == "for")
  return tok_for;
if (IdentifierStr == "in")
  return tok_in;
return tok_identifier;
```

### AST Extensions for the 'for' Loop

The AST node is just as simple. It basically boils down to capturing the
variable name and the constituent expressions in the node.

```cpp
/// ForExprAST - Expression class for for/in.
class ForExprAST : public ExprAST {
  std::string VarName;
  std::unique_ptr<ExprAST> Start, End, Step, Body;

public:
  ForExprAST(const std::string &VarName, std::unique_ptr<ExprAST> Start,
             std::unique_ptr<ExprAST> End, std::unique_ptr<ExprAST> Step,
             std::unique_ptr<ExprAST> Body)
    : VarName(VarName), Start(std::move(Start)), End(std::move(End)),
      Step(std::move(Step)), Body(std::move(Body)) {}

  Value codegen() override;
};
```

### Parser Extensions for the 'for' Loop

The parser code is also fairly standard. The only interesting thing here
is handling of the optional step value. The parser code handles it by
checking to see if the second comma is present. If not, it sets the step
value to null in the AST node:

```cpp
/// forexpr ::= 'for' identifier '=' expr ',' expr (',' expr)? 'in' expression
static std::unique_ptr<ExprAST> ParseForExpr() {
  getNextToken();  // eat the for.

  if (CurTok != tok_identifier)
    return LogError("expected identifier after for");

  std::string IdName = IdentifierStr;
  getNextToken();  // eat identifier.

  if (CurTok != '=')
    return LogError("expected '=' after for");
  getNextToken();  // eat '='.


  auto Start = ParseExpression();
  if (!Start)
    return nullptr;
  if (CurTok != ',')
    return LogError("expected ',' after for start value");
  getNextToken();

  auto End = ParseExpression();
  if (!End)
    return nullptr;

  // The step value is optional.
  std::unique_ptr<ExprAST> Step;
  if (CurTok == ',') {
    getNextToken();
    Step = ParseExpression();
    if (!Step)
      return nullptr;
  }

  if (CurTok != tok_in)
    return LogError("expected 'in' after for");
  getNextToken();  // eat 'in'.

  auto Body = ParseExpression();
  if (!Body)
    return nullptr;

  return std::make_unique<ForExprAST>(IdName, std::move(Start),
                                       std::move(End), std::move(Step),
                                       std::move(Body));
}
```

And again we hook it up as a primary expression:

```cpp
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
  }
}
```

### MLIR and LLVM IR for the 'for' Loop

Now we get to the good part: the MLIR we want to generate for this
construct. With the simple example above, we get:

```mlir
func.func private @putchard(f64) -> f64

func.func @printstar(%arg0: f64) -> f64 {
  %cst = arith.constant 4.200000e+01 : f64
  %cst_0 = arith.constant 0.000000e+00 : f64
  %cst_1 = arith.constant 1.000000e+00 : f64
  %0 = scf.while (%arg1 = %cst_1) : (f64) -> f64 {
    %1 = arith.cmpf ult, %arg1, %arg0 : f64
    scf.condition(%1) %arg1 : f64
  } do {
  ^bb0(%arg1: f64):
    %1 = func.call @putchard(%cst) : (f64) -> f64
    %2 = arith.addf %arg1, %cst_1 : f64
    scf.yield %2 : f64
  }
  return %cst_0 : f64
}
```

The loop is represented by an
[`scf.while`](https://mlir.llvm.org/docs/Dialects/SCFDialect/#scfwhile-scfwhileop)
operation. Its loop-carried value, `%arg1`, is the current value of the
induction variable. It begins with `%cst_1`, which is `1.0` in this
example.

The first region tests the end condition before each iteration. The
[`scf.condition`](https://mlir.llvm.org/docs/Dialects/SCFDialect/#scfcondition-scfconditionop)
operation determines whether the loop continues and forwards the current
induction value to the `do` region. Consequently, a false initial
condition prevents the body from running at all. The `do` region emits
the body and step, then uses
[`scf.yield`](https://mlir.llvm.org/docs/Dialects/SCFDialect/#scfyield-scfyieldop)
to carry the next induction value back to the condition.

When this structured control flow is lowered, the loop-carried region
argument becomes a PHI node. The automatically numbered names in the
actual output have been replaced with descriptive names below for
clarity:

```llvm
declare double @putchard(double)

define double @printstar(double %n) {
entry:
  br label %loop

loop:                                             ; preds = %body, %entry
  %i = phi double [ %nextvar, %body ], [ 1.000000e+00, %entry ]
  %loopcond = fcmp ult double %i, %n
  br i1 %loopcond, label %body, label %afterloop

body:                                             ; preds = %loop
  %bodyi = phi double [ %i, %loop ]
  %calltmp = call double @putchard(double 4.200000e+01)
  %nextvar = fadd double %bodyi, 1.000000e+00
  br label %loop

afterloop:                                        ; preds = %loop
  ret double 0.000000e+00
}
```

This loop contains the same basic blocks and PHI nodes that we saw in the
lowered if/then/else expression. The PHI node selects `1.0` when control
first enters the loop from `entry`, and `%nextvar` when control returns
along the loop backedge. The condition is tested before branching to
`body`. In the MLIR above, the `scf.while` region arguments express
these relationships directly.

### Code Generation for the 'for' Loop

The first part of codegen is very simple: we emit the start expression
before putting the loop variable in scope:

```cpp
Value ForExprAST::codegen() {
  // Emit the start value before putting the loop variable in scope.
  Value StartVal = Start->codegen();
  if (!StartVal)
    return {};
```

Next, we save any existing symbol with the same name as the loop
variable:

```cpp
  auto OldValue = NamedValues.find(VarName);
  bool HadOldValue = OldValue != NamedValues.end();
  Value SavedValue = HadOldValue ? OldValue->second : Value();
  bool CodegenFailed = false;
```

MLIR regions define the scope of their SSA values, but they do not
automatically manage the `NamedValues` map used by our frontend. We
still need that map to resolve a source-level name such as `i` while
walking the AST. Saving its previous entry allows a loop variable to
shadow a function argument or an enclosing loop variable without making
the outer value inaccessible after the loop.

We can now create the `scf.while` operation. Its initial loop-carried
value is `StartVal`. The first region evaluates the condition before the
body is entered:

```cpp
  // The "before" region tests the loop condition. The "after" region emits
  // the body and step, then carries the next induction value back to be tested.
  TheBuilder->create<scf::WhileOp>(
      getLocation(), TypeRange{TheBuilder->getF64Type()},
      ValueRange{StartVal},
      [&](OpBuilder &Builder, Location Loc, ValueRange Args) {
        NamedValues[VarName] = Args.front();

        Value EndCond = End->codegen();
        if (!EndCond) {
          CodegenFailed = true;
          EndCond = Builder.create<arith::ConstantOp>(
              Loc, Builder.getF64FloatAttr(0.0));
        }

        Value Zero = Builder.create<arith::ConstantOp>(
            Loc, Builder.getF64FloatAttr(0.0));
        EndCond = Builder.create<arith::CmpFOp>(
            Loc, arith::CmpFPredicate::ONE, EndCond, Zero);
        Builder.create<scf::ConditionOp>(Loc, EndCond, Args.front());
      },
```

`Args.front()` is the current SSA value of the induction variable. We
enter it in `NamedValues` so that the end expression can refer to the
loop variable. The `scf.condition` operation enters the second region
only when the condition is true, forwarding the current induction value
to it.

The second region emits the body and calculates the next value of the
induction variable by adding the step expression, or `1.0` when no step
was specified:

```cpp
      [&](OpBuilder &Builder, Location Loc, ValueRange Args) {
        NamedValues[VarName] = Args.front();

        if (!Body->codegen())
          CodegenFailed = true;

        Value StepVal;
        if (Step)
          StepVal = Step->codegen();
        else
          StepVal = Builder.create<arith::ConstantOp>(
              Loc, Builder.getF64FloatAttr(1.0));
        if (!StepVal) {
          CodegenFailed = true;
          StepVal = Builder.create<arith::ConstantOp>(
              Loc, Builder.getF64FloatAttr(1.0));
        }

        Value NextVar =
            Builder.create<arith::AddFOp>(Loc, Args.front(), StepVal);
        Builder.create<scf::YieldOp>(Loc, NextVar);
      });
```

The `scf.yield` operation carries `NextVar` back to the first region,
where the condition is evaluated again. MLIR handles the blocks and
their arguments, so we do not need to construct the loop's PHI nodes or
backedge ourselves.

After constructing the loop, we restore the source-level symbol that was
shadowed, or remove the loop variable if no previous definition existed:

```cpp
  // Restore any variable shadowed by the loop induction variable.
  if (HadOldValue)
    NamedValues[VarName] = SavedValue;
  else
    NamedValues.erase(VarName);

  if (CodegenFailed)
    return {};

  // A for expression always returns 0.0.
  return TheBuilder->create<arith::ConstantOp>(
      getLocation(), TheBuilder->getF64FloatAttr(0.0));
}
```

The generated SSA value remains scoped to the `scf.while` regions, while
restoring `NamedValues` keeps the frontend's view of source-level scope
in sync. Finally, code generation of the for loop always returns `0.0`.

With this, we conclude the "adding control flow to Kaleidoscope" chapter
of the tutorial. In this chapter we added two control flow constructs,
and used them to motivate a couple of aspects of the LLVM IR that are
important for front-end implementors to know. In the next chapter of our
saga, we will get a bit crazier and add [user-defined
operators](chapter-06.md) to our poor innocent language.

## Control Flow Graph Visualization Tools

To visualize the control flow graph, you can use MLIR's
`--view-op-graph` option. The output above shows one REPL interaction at a
time, so it is not quite a standalone MLIR file. Save the complete module below
as `t.mlir`:

```mlir
module {
  func.func private @foo() -> f64
  func.func private @bar() -> f64

  func.func @baz(%arg0: f64) -> f64 {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = arith.cmpf one, %arg0, %cst : f64
    %1 = scf.if %0 -> (f64) {
      %2 = func.call @foo() : () -> f64
      scf.yield %2 : f64
    } else {
      %2 = func.call @bar() : () -> f64
      scf.yield %2 : f64
    }
    return %1 : f64
  }
}
```

Then run:

```bash
mlir-opt \
  t.mlir \
  --convert-scf-to-cf \
  '--view-op-graph=print-control-flow-edges' \
  -o /dev/null 2>&1 \
  | perl -pe 's/style = filled/style = solid/g; s/fillcolor = "[^"]+"/fillcolor = "transparent"/g' \
  > t-mlir.dot

dot -Tsvg -Gbgcolor=transparent \
  -Gcolor=black -Gfontcolor=black \
  -Ncolor=black -Nfontcolor=black \
  -Ecolor=black -Efontcolor=black \
  -o t-mlir.svg t-mlir.dot

dot -Tsvg -Gbgcolor=transparent \
  -Gcolor=gray -Gfontcolor=gray \
  -Ncolor=gray -Nfontcolor=gray \
  -Ecolor=gray -Efontcolor=gray \
  -o t-mlir-gray.svg t-mlir.dot
```

The `--convert-scf-to-cf` pass first lowers `scf.if` into basic blocks
connected by `cf.cond_br` and `cf.br` operations.
`--view-op-graph=print-control-flow-edges` then writes a Graphviz
representation of the operations, blocks, data-flow edges, and control-flow
edges. The remaining commands remove the default node colors and render two
graphs with transparent backgrounds: a black version for light mode and a gray
version that remains visible in dark mode.

On macOS, you can open the resulting graph with:

```bash
open t-mlir.svg
```

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="images/t-mlir-gray.svg">
  <img src="images/t-mlir.svg" alt="MLIR control-flow graph">
</picture>

To visualize the control flow graph for LLVM IR, you can use a nifty feature of the LLVM '[opt](https://llvm.org/cmds/opt.html)' tool. If you put this LLVM IR into "t.ll" and run "`llvm-as < t.ll | opt -passes=view-cfg`", [a window will pop up](https://llvm.org/docs/ProgrammersManual.html#viewing-graphs-while-debugging-code) and you'll see this graph:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="images/t-llvm-gray.svg">
  <img src="images/t-llvm.svg" alt="LLVM control-flow graph">
</picture>

## Full Code Listing

Here is the complete code listing for our running example, enhanced with
the if/then/else and for expressions. Here is the CMake configuration:

```cmake(../code/chapter-05/CMakeLists.txt)
```

To build this example, use:

```bash
cmake -S . -B build \
  -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir
cmake --build build
./build/toy
```

Here is the code:

```cpp(../code/chapter-05/toy.cpp)
```

[Next: Extending the language: user-defined operators](chapter-06.md)
