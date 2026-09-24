# Kaleidoscope with MLIR

This is an attempt to rebuild LLVM's little Kaleidoscope language using MLIR,
without turning it into an entirely different tutorial along the way.

The starting point is LLVM's excellent
[My First Language Frontend with LLVM](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/).
It begins with a lexer, grows an AST, generates some IR, adds a JIT, and keeps
going until the toy language feels surprisingly real. This tutorial follows
roughly the same walk. The syntax is familiar, most of the examples are
familiar, and the chapters arrive in more or less the same order.

The difference is that we do not generate LLVM IR immediately. We spend some
time in MLIR first.

That makes this tutorial a loose bridge between the LLVM Kaleidoscope tutorial
and MLIR's [Toy tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/). It starts
on the Kaleidoscope side, where the compiler is small and direct, and slowly
drifts toward the MLIR way of thinking: dialects, regions, progressive
lowering, conversion passes, and eventually a tiny dialect of our own.

Toy goes much further. It begins with a language-specific dialect and uses it
to explore tensors, shape inference, rewrites, and progressively lower-level
representations. We do not try to duplicate that here. The hope is simply that
after finishing this tutorial, Toy feels like the next step rather than a
different universe.

## The Pipeline

The compiler eventually looks something like this:

```text
Kaleidoscope source
        ↓
lexer, parser, and AST
        ↓
Kaleidoscope and standard MLIR dialects
        ↓
LLVM dialect
        ↓
LLVM IR
        ↓
JIT-compiled code or an object file
```

MLIR is not replacing LLVM in that picture. It gives us a few useful places to
stand before LLVM takes over.

## Drifts From The Original Kaleidoscope LLVM tutorial

### [Chapter 1: The Lexer](docs/chapter-01.md)

This is still almost entirely Kaleidoscope territory. We introduce the
language and write the same sort of tiny hand-made lexer as the LLVM tutorial.
There is barely any MLIR to worry about yet, which is probably for the best.

### [Chapter 2: The Parser and AST](docs/chapter-02.md)

Again, this stays close to the original route: recursive descent parsing,
operator precedence, and a small tree of expression classes.

### [Chapter 3: Generating MLIR](docs/chapter-03.md)

The original tutorial generates LLVM IR from the AST. We generate operations from the `arith` and `func`
dialects instead. Functions contain regions, regions contain blocks, and the IR still looks pleasantly close to the source program.

We also start using `--dump-mlir`, because being able to look at each function as it is created makes the rest of the tutorial much easier to follow.

### [Chapter 4: Optimization and the JIT](docs/chapter-04.md)

The broad idea is still borrowed from Kaleidoscope: optimize the code and make
it run interactively. The path is different, though. MLIR canonicalization and
CSE happen before we lower through the LLVM dialect, translate to LLVM IR, and
hand the result to ORC.

This is also where `--dump-llvm-ir` appears. From here onward we can look at
both MLIR and LLVM IR.

### [Chapter 5: Control Flow](docs/chapter-05.md)

We begin higher up with the `scf` dialect, where an `if` really is an
`if` with regions and yielded values. Lowering turns that into the `cf`
dialect, region results become block arguments, and those block arguments
eventually become LLVM PHI nodes.

### [Chapter 6: User-Defined Operators](docs/chapter-06.md)

This is very similar to the original LLVM approach to defining unary and binary operators. The generated
calls and arithmetic are still MLIR operations.

### [Chapter 7: Mutable Variables](docs/chapter-07.md)

Mutation brings the SSA question back. Rather than teaching the frontend to manufacture PHI nodes, we represent variables as memory, use loads and stores, and let the pass pipeline promote that memory back into SSA. We can watch the output move through structured MLIR, explicit control flow, block arguments, and finally LLVM IR.

### [Chapter 8: Object Files](docs/chapter-08.md)

Dropping down to LLVM now takes over for the target-specific work. We translate to LLVM IR, choose a target, emit an object
file, and link a small native program. The chapter also supports selecting a
target triple and choosing the output filename, mostly because it is useful to
see where target independence actually ends.

### [Chapter 9: Debug Information](docs/chapter-09.md)

We build a small custom MLIR pass that adds the language-specific compile-unit,
function, and variable information before translation turns source locations into LLVM debug
metadata and, eventually, DWARF.

This is our first custom pass. The implementation is slightly fiddly because
debug information is slightly fiddly, but the chapter can mostly treat it as a
useful compiler pass and keep moving.

### [Chapter 10: A Small Kaleidoscope Dialect](docs/chapter-10.md)

This is where the tutorial leans most clearly toward Toy. We do not lift the
whole language into a custom dialect; that would more or less begin another
tutorial. Instead, we preserve just enough source-level meaning for mutable
variables.

The dialect knows that a value is a variable named `x` or `y`, rather than an
anonymous piece of storage. Its lowering creates the allocation and debug
declaration together, while that information is still easy to understand. It
is a small example, but it gets at the reason custom dialects exist.

### [Chapter 11: Where to Go Next](docs/chapter-11.md)

The conclusion keeps the original spirit of "go add something strange to the
language," but now there are two directions to wander. You can continue down
into LLVM, native code generation, runtimes, and ABI details, or move upward
into richer dialects, more passes, and a fuller MLIR representation of the
language.

## Building the Chapters

Each `code/chapter-XX` directory is a snapshot of the compiler at that point in
the tutorial. The later chapters use CMake and include a small `build.sh` for
local convenience. You will need a build of LLVM with MLIR available; the
chapter text shows the relevant commands when the build changes.

The documentation lives in `docs/`, and the matching implementation lives in
`code/`.

## A Small Disclaimer

This follows the structure and many of the ideas of LLVM's Kaleidoscope
tutorial, and it owes an obvious debt to the MLIR Toy tutorial too. It is an
independent experiment, not official LLVM documentation.
