# 11. Kaleidoscope: Conclusion and other useful MLIR tidbits

## Tutorial Conclusion

Welcome to the final chapter of the "[Implementing a language with
MLIR](chapter-00.md)" tutorial. In the course of this tutorial, we have
grown our little Kaleidoscope language from being a useless toy, to
being a semi-interesting (but probably still useless) toy. :)

It is interesting to see how far we've come, and how little code it has
taken. We built the entire lexer, parser, and AST, generated MLIR
through the `arith` and `func` dialects, ran real MLIR optimization
passes, added an interactive run-loop backed by a genuine incremental
ORC JIT, lowered progressively through the `scf` and `llvm` dialects to
object files, and emitted debug information for standalone executables
- all in a bit over 1,000 lines of non-comment, non-blank code.

Our little language supports a couple of interesting features: it
supports user defined binary and unary operators, it uses JIT
compilation for immediate evaluation, and it supports a few control flow
constructs with SSA construction. Because we stayed in MLIR's dialects
as long as possible, all of that is visible as readable, structured IR
right up until the point it has to become machine code.

Part of the idea of this tutorial was to show you how easy and fun it
can be to define, build, and play with languages using MLIR. Building a
compiler need not be a scary or mystical process! Now that you've seen
some of the basics, I strongly encourage you to take the code and hack
on it. For example, try adding:

- **global variables** - While global variables have questionable value
  in modern software engineering, they are often useful when putting
  together quick little hacks like the Kaleidoscope compiler itself.
  Fortunately, our current setup makes it very easy to add global
  variables: just have value lookup check to see if an unresolved
  variable is in the global variable symbol table before rejecting it.
  MLIR's `memref` dialect gives you a ready-made way to do this: create
  a module-level symbol with `memref.global`, and access it from
  anywhere in the module with `memref.get_global`.
- **typed variables** - Kaleidoscope currently only supports variables
  of type double. This gives the language a very nice elegance, because
  only supporting one type means that you never have to specify types.
  Different languages have different ways of handling this. The easiest
  way is to require the user to specify types for every variable
  definition, and record the type of the variable in the symbol table
  alongside its MLIR `Value`. Because MLIR's type system isn't limited
  to what a single dialect provides, you aren't restricted to `f64` and
  a handful of primitives the way you would be with a fixed instruction
  set - any registered dialect's types are available to you.
- **arrays, structs, vectors, etc** - Once you add types, you can start
  extending the type system in all sorts of interesting ways. MLIR gives
  you two reasonable paths here. If you stay at the `memref`/`tensor`
  level, indexing is structural - `memref.load` and `memref.store` take
  SSA index values directly, no separate address-computation instruction
  required. If you'd rather work closer to the metal, once you're in the
  `llvm` dialect you have direct access to
  [`llvm.getelementptr`](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmgetelementptr-llvmgepop),
  MLIR's mirror of LLVM's `getelementptr` instruction: it is so
  nifty/unconventional, it [has its own
  FAQ](https://llvm.org/docs/GetElementPtr.html)!
- **standard runtime** - Our current language allows the user to access
  arbitrary external functions, and we use it for things like "printd"
  and "putchard". As you extend the language to add higher-level
  constructs, often these constructs make the most sense if they are
  lowered to calls into a language-supplied runtime. For example, if
  you add hash tables to the language, it would probably make sense to
  add the routines to a runtime, instead of inlining them all the way.
- **memory management** - Currently we can only access the stack in
  Kaleidoscope. It would also be useful to be able to allocate heap
  memory, either with calls to the standard libc malloc/free interface
  or with a garbage collector. If you would like to use garbage
  collection, note that MLIR's `llvm` dialect carries the same GC
  support LLVM IR does: `llvm.func` accepts a `garbageCollector`
  attribute naming the GC strategy, which flows straight through to
  LLVM's [Accurate Garbage
  Collection](https://llvm.org/docs/GarbageCollection.html) machinery,
  including algorithms that move objects and need to scan/update the
  stack.
- **exception handling support** - Once lowered to the `llvm` dialect,
  MLIR gives you the same building blocks LLVM IR does for [zero cost
  exceptions](https://llvm.org/docs/ExceptionHandling.html) -
  `llvm.invoke`, `llvm.landingpad`, and `llvm.resume` all exist as real
  MLIR operations that interoperate with code compiled in other
  languages. You could also generate code by implicitly making every
  function return an error value and checking it, or make explicit use
  of setjmp/longjmp. There are many different ways to go here.
- **object orientation, generics, database access, complex numbers,
  geometric programming, ...** - Really, there is no end of crazy
  features that you can add to the language.
- **unusual domains** - We've been talking about applying MLIR to a
  domain that many people are interested in: building a compiler for a
  specific language. However, MLIR's whole premise is that compiler
  infrastructure built for one domain should be reusable in others - the
  same dialect-and-progressive-lowering architecture you just used for
  Kaleidoscope is what powers machine learning compilers like XLA and
  IREE, and hardware design tools like CIRCT. Maybe you will be the
  first to JIT compile a regular expression interpreter into native code
  with MLIR?

Have fun - try doing something crazy and unusual. Building a language
like everyone else always has, is much less fun than trying something a
little crazy or off the wall and seeing how it turns out. If you get
stuck or want to talk about it, please post on the [LLVM
forums](https://discourse.llvm.org) or the [MLIR
Discourse](https://discourse.llvm.org/c/mlir/31): both have lots of
people who are interested in languages and are often willing to help
out.

If you'd rather keep going with Kaleidoscope itself, [Chapter
10](chapter-10.md) is where this tutorial stops mapping LLVM concepts
one-to-one and starts using MLIR for what it's actually for: it grows
Kaleidoscope's own operators into a small custom dialect, showing how a
language can carry its own semantics through the early passes instead
of committing to `arith`/`func` immediately. That's the same idea MLIR's
[Toy tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/) takes much
further with tensors and shape inference - if the custom-dialect chapter
was interesting to you, Toy is the natural next stop.

Before we end this tutorial, I want to talk about some "tips and tricks"
for generating MLIR. These are some of the more subtle things that
may not be obvious, but are very useful if you want to take advantage of
MLIR's capabilities.

## Properties of MLIR

We have a couple of common questions about code in MLIR's form -
let's just get these out of the way right now, shall we?

### Target Independence

Kaleidoscope is an example of a "portable language": any program written
in Kaleidoscope will work the same way on any target that it runs on.
Many other languages have this property, e.g. lisp, java, haskell,
javascript, python, etc (note that while these languages are portable,
not all their libraries are).

MLIR takes target independence further than a single-level IR like LLVM
IR can. Because Kaleidoscope stayed in `arith`, `func`, and `scf` for as
long as possible and only converted to the `llvm` dialect right before
execution, the same high-level IR isn't committed to becoming LLVM IR at
all - it could just as easily be lowered toward a completely different
backend (a GPU dialect, SPIR-V, and so on) without touching the front
end. You can trivially tell that the Kaleidoscope compiler generates
target-independent code up through that point because it never queries
for any target-specific information when generating the high-level
dialects; target-specific concerns like the data layout only enter the
picture in `lowerToLLVM`, right where we hand a module to the JIT.

The fact that MLIR (like LLVM) provides a compact, target-independent
representation for code gets a lot of people excited. Unfortunately,
these people are usually thinking about C or a language from the C
family when they are asking questions about language portability. I say
"unfortunately", because there is really no way to make (fully general)
C code portable, other than shipping the source code around (and of
course, C source code is not actually portable in general either - ever
port a really old application from 32- to 64-bits?).

The problem with C (again, in its full generality) is that it is heavily
laden with target-specific assumptions. As one simple example, the
preprocessor often destructively removes target-independence from the
code when it processes the input text:

```c
#ifdef __i386__
  int X = 1;
#else
  int X = 42;
#endif
```

While it is possible to engineer more and more complex solutions to
problems like this, it cannot be solved in full generality in a way that
is better than shipping the actual source code.

That said, there are interesting subsets of C that can be made portable.
If you are willing to fix primitive types to a fixed size (say int =
32-bits, and long = 64-bits), don't care about ABI compatibility with
existing binaries, and are willing to give up some other minor features,
you can have portable code. This can make sense for specialized domains
such as an in-kernel language.

### Safety Guarantees

Many of the languages above are also "safe" languages: it is impossible
for a program written in Java to corrupt its address space and crash the
process (assuming the JVM has no bugs). Safety is an interesting
property that requires a combination of language design, runtime
support, and often operating system support.

Whether MLIR gives you that guarantee depends on which dialect you're
in. The dialects Kaleidoscope uses through most of this tutorial -
`arith`, `func`, `scf` - only let you build well-typed SSA values; there
is no operation in those dialects for an unsafe pointer cast or an
out-of-bounds access. That property goes away the moment you convert to
the `llvm` dialect: it allows unsafe pointer casts, use after free bugs,
buffer over-runs, and the same variety of problems raw LLVM IR does,
because at that point it effectively *is* LLVM IR. Safety needs to be
implemented as a layer on top, and, conveniently, several groups have
investigated this both for LLVM and for MLIR. Ask on the [LLVM
forums](https://discourse.llvm.org) if you are interested in more
details.

### Language-Specific Optimizations

One thing about compiler infrastructure that turns off many people is
that it doesn't solve all the world's problems in one system. One
specific complaint people have about LLVM is that it is perceived as
being incapable of performing high-level language-specific optimization:
LLVM "loses too much information" once everything is lowered to a single
fixed instruction set.

MLIR's answer to this is structurally different from LLVM's, and it's
the thing this tutorial's later chapters are really about: instead of
asking you to bolt language-specific passes onto a fixed IR, MLIR lets
you define your own operations, types, and passes, exactly as
[Chapter 10](chapter-10.md) does for Kaleidoscope's operators. A
`kaleidoscope.binary` operation is not an `arith.mulf` wearing a
disguise - it genuinely carries information ("this came from operator
syntax, this is the `%` operator") that a lowered form can't recover.
You get to decide how long that information survives before you convert
it away, and you can write real canonicalization and rewrite patterns
against it in the meantime.

This doesn't mean MLIR is immune to the same tradeoff LLVM faces once
you *do* lower: the `arith` and `llvm` dialects use structural type
equivalence just like LLVM IR does, so two high-level types that
happen to lower to the same `f64` or the same `!llvm.struct<(i32)>`
become indistinguishable once you're there (other than debug info). The
difference is that MLIR lets you choose, dialect by dialect, how much of
that structure to keep before you pay that cost - and, just as with
LLVM, if you have a specific need and run into a wall, the [MLIR
Discourse](https://discourse.llvm.org/c/mlir/31) is a good place to ask.
At the very worst, you can always treat any dialect as if it were a
"dumb code generator" and implement the high-level optimizations you
desire in your front-end, on the language-specific AST - exactly what
Kaleidoscope's own `arith.constant` folding did for free back in
[Chapter 4](chapter-04.md#trivial-constant-folding).

## Tips and Tricks

There is a variety of useful tips and tricks that you come to know after
working on/with MLIR that aren't obvious at first glance. Instead of
letting everyone rediscover them, this section talks about some of these
issues.

### Implementing portable offsetof/sizeof

One interesting thing that comes up, if you are trying to keep the code
generated by your compiler "target independent", is that you often need
to know the size of some type or the offset of some field in a
structure. For example, you might need to pass the size of a type into a
function that allocates memory.

Unfortunately, this can vary widely across targets: for example the
width of a pointer is trivially target-specific. If you're already
working in the `llvm` dialect, the same [clever way to use the
getelementptr
instruction](http://nondot.org/sabre/LLVMNotes/SizeOf-OffsetOf-VariableSizedStructs.txt)
that LLVM IR uses works unchanged, since `llvm.getelementptr` is the
same operation under a different syntax. But MLIR also gives you a more
direct, dialect-independent answer that doesn't require the GEP trick at
all: `mlir::DataLayout` exposes `getTypeSize` and `getTypeSizeInBits`
for any type, resolved against whichever target data layout you're
compiling for - the same data layout chapter 4's JIT already asks the
`ExecutionEngine`/`KaleidoscopeJIT` for before translating to LLVM IR.

### Garbage Collected Stack Frames

Some languages want to explicitly manage their stack frames, often so
that they are garbage collected or to allow easy implementation of
closures. There are often better ways to implement these features than
explicit stack frames, but once you're in the `llvm` dialect, MLIR
carries the same support LLVM does for this
(<http://nondot.org/sabre/LLVMNotes/ExplicitlyManagedStackFrames.txt>),
via the same `garbageCollector` attribute mentioned above. It requires
your front-end to convert the code into [Continuation Passing
Style](http://en.wikipedia.org/wiki/Continuation-passing_style) and
the use of tail calls (which MLIR's `llvm` dialect also supports).
