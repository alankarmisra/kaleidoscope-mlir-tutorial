# Chapter 11: Conclusion and Other Useful MLIR Tidbits

## You Made It!

Welcome to the final chapter of the [Implementing a Language with MLIR](chapter-00.md) tutorial. Take a moment to appreciate what you've built. You started with a bare lexer and parser, and you've ended up with a real compiler that:

- Lexes, parses, and builds an AST for a working language
- Generates MLIR through standard dialects like `arith`, `func`, and `scf`
- Runs actual MLIR optimization passes on that IR
- Executes code interactively through an incremental ORC JIT
- Lowers progressively down through the `llvm` dialect to object files
- Emits DWARF debug information for standalone executables

And it's all in a compact set of source files you can read, understand, and change.

Along the way, your language picked up some genuinely interesting features: user-defined binary and unary operators, JIT compilation for immediate evaluation, and control flow constructs with proper SSA construction. The reason all of that stayed readable is that we kept the program in MLIR's high-level dialects for as long as possible, only dropping to machine code at the very end.

The bigger lesson here is that **building a compiler doesn't have to be scary or mystical.** MLIR gives you clear, well-defined levels to work at, and you can move between them one step at a time.

## The Pipeline You Built

Here's the whole journey in one picture:

```text
Kaleidoscope source
        ↓
Kaleidoscope AST
        ↓
Kaleidoscope and standard MLIR dialects
        ↓
LLVM dialect
        ↓
LLVM IR
        ↓
JIT-compiled code or an object file
```

Notice what this picture *doesn't* say. MLIR hasn't replaced LLVM. Each level has a job. MLIR gives your frontend a clean way to express what your language means. LLVM handles optimization, code generation, and target support once you're ready to commit to real hardware.

## Ideas for What to Build Next

Now that you have a working compiler, the best thing you can do is **hack on it.** Here are some directions that our current setup makes surprisingly easy:

**Global variables.** These are questionable in modern software engineering, but they're great for quick little hacks. Our value lookup already has a natural place to add them: before rejecting an unresolved variable, check a global symbol table. MLIR makes this easy with `memref.global` for the declaration and `memref.get_global` for the access.

**Typed variables.** Kaleidoscope currently only has `double`. That's a nice simplification because you never have to write a type — but it's also limiting. The easiest extension is to require type annotations on variable definitions and store the type alongside the `Value` in your symbol table. The nice part: MLIR's type system isn't limited to a fixed set of primitives. Any registered dialect's types are available, so you're not stuck with just `f64`.

**Arrays, structs, vectors, and friends.** Once you add types, the type system opens up. MLIR gives you two paths here:

- Stay at the `memref`/`tensor` level, where indexing is structural — `memref.load` and `memref.store` take SSA indices directly, with no separate address computation step.
- Drop to the `llvm` dialect, where you have direct access to [`llvm.getelementptr`](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmgetelementptr-llvmgepop), MLIR's mirror of LLVM's famous `getelementptr`. It's so unconventional it [has its own FAQ](https://llvm.org/docs/GetElementPtr.html).

**A standard runtime.** Right now, users can call any external function — which is how `printd` and `putchard` work. As you add higher-level features, they'll often make more sense as calls into a language runtime. Adding hash tables to the language? Those routines probably belong in a runtime library, not inlined at every call site.

**Memory management.** Today, everything lives on the stack. You'd probably want heap allocation, either through libc's `malloc`/`free` or a garbage collector. If you go the GC route, good news: MLIR's `llvm` dialect carries the same GC support LLVM IR does. The `llvm.func` operation accepts a `garbageCollector` attribute that names a GC strategy, and it flows straight into LLVM's [Accurate Garbage Collection](https://llvm.org/docs/GarbageCollection.html) machinery — including moving collectors that need to scan and update the stack.

**Exception handling.** Once you're in the `llvm` dialect, MLIR gives you the same building blocks LLVM IR does for [zero-cost exceptions](https://llvm.org/docs/ExceptionHandling.html): `llvm.invoke`, `llvm.landingpad`, and `llvm.resume` are all real operations that interoperate with code compiled in other languages. You could also go simpler — every function returns an error value, or you use `setjmp`/`longjmp`. There's no single right answer.

**Object orientation, generics, database access, complex numbers, geometric programming…** Really, there's no end to what you could add.

**Something completely unusual.** Most of this tutorial has been about a familiar domain — building a compiler for a specific language. But MLIR's whole premise is that compiler infrastructure built for one domain should be reusable in others. The same dialect-and-progressive-lowering architecture you just used for Kaleidoscope powers machine learning compilers like XLA and IREE, and hardware design tools like CIRCT. Maybe you'll be the first to JIT-compile a regex interpreter into native code.

**Go do something weird.** Building a language the way everyone else has is far less fun than trying something slightly crazy and seeing how it turns out. If you get stuck or want to talk it through, the [LLVM forums](https://discourse.llvm.org) and the [MLIR Discourse](https://discourse.llvm.org/c/mlir/31) both have plenty of people who enjoy this stuff and are happy to help.

## Where to Go from Here

If you'd rather keep going with compiler architecture, there are several natural directions:

- **Grow the Kaleidoscope dialect.** Chapter 10 introduced it for variables only. The next step would be representing the complete source program in language-specific MLIR. The [Toy tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/) develops this design much further, with tensors and shape inference.
- **Add verifiers, traits, canonicalization patterns, and dialect interfaces.** These let your dialect define what valid IR means and teach generic MLIR infrastructure how your operations behave. See the docs on [canonicalization](https://mlir.llvm.org/docs/Canonicalization/) and [interfaces](https://mlir.llvm.org/docs/Interfaces/).
- **Introduce more lowering stages.** A larger compiler doesn't have to jump straight from source dialect to LLVM. It can lower progressively through whichever standard or project-specific dialects express each intermediate form best.
- **Write more custom passes.** Chapter 9 used one to construct debug scopes. Chapter 10 used [dialect conversion](https://mlir.llvm.org/docs/DialectConversion/) to lower variables. The same [pass infrastructure](https://mlir.llvm.org/docs/PassManagement/) can implement language-specific analysis and optimization while the relevant semantics are still present.
- **Target something other than LLVM.** Operations that stay in higher-level, target-independent dialects could be lowered toward GPU, SPIR-V, or another backend without touching the parser or AST.

LLVM remains the natural direction when you want better native code generation: inspect the translated LLVM IR, add LLVM optimization pipelines, extend the ORC JIT, handle more ABI details, or integrate a runtime and system libraries.

The real design question isn't "MLIR or LLVM?" It's "where should each source-language concept be lowered from one level to the next?"

---

Before we wrap up, let's talk about some subtle but very useful properties of MLIR — things that aren't obvious at first but pay off once you know them.

## Properties of MLIR

### Target Independence

Kaleidoscope is a **portable language**: any program written in it behaves the same way on any target it runs on. Many other languages share this property — Lisp, Java, Haskell, JavaScript, Python — though it's worth noting that while these *languages* are portable, not all their libraries are.

MLIR takes target independence further than a single-level IR like LLVM IR can. Because Kaleidoscope stays in `arith`, `func`, and `scf` for as long as possible and only converts to the `llvm` dialect right before execution, that high-level IR isn't committed to becoming LLVM IR at all. It could just as easily be lowered toward a GPU dialect, SPIR-V, or a completely different backend without touching the frontend.

You can see this directly in the code: the compiler never queries for target-specific information while generating the high-level dialects. Target-specific concerns like data layout only enter the picture inside `lowerToLLVM`, right where the module gets handed to the JIT.

Now, a lot of people get excited about this and immediately think about C. Unfortunately, they're usually thinking about C or a C-family language, and that's a problem — because there's really no way to make fully general C code portable other than shipping the source code around. And even then, C source isn't actually portable in general (ever try to port a really old application from 32-bit to 64-bit?).

The issue is that C, in full generality, is heavily laden with target-specific assumptions. A simple example: the preprocessor often destroys target independence just by running:

```c
#ifdef __i386__
  int X = 1;
#else
  int X = 42;
#endif
```

You can engineer increasingly complex solutions to problems like this, but you can't solve it in full generality in a way that beats shipping the actual source.

That said, there are useful *subsets* of C that can be made portable. If you fix primitive types to specific sizes (say, `int` = 32 bits, `long` = 64 bits), give up ABI compatibility with existing binaries, and sacrifice a few other minor features, you can have portable code. That can make sense for specialized domains like an in-kernel language.

### Safety Guarantees

Many of the languages listed above are also **safe** languages: it's impossible for a program written in Java to corrupt its address space and crash the process (assuming the JVM has no bugs). Safety is an interesting property — it needs language design, runtime support, and often operating system support all working together.

Whether MLIR gives you that guarantee depends on which dialect you're in. The dialects Kaleidoscope uses through most of the tutorial — `arith`, `func`, `scf` — only let you build well-typed SSA values. There's no operation in those dialects for an unsafe pointer cast or an out-of-bounds access.

That property disappears the moment you convert to the `llvm` dialect. It allows unsafe pointer casts, use-after-free bugs, buffer overruns, and the same variety of problems raw LLVM IR has — because at that point, it *effectively is* LLVM IR. Safety has to be implemented as a layer on top, and several groups have investigated exactly that for both LLVM and MLIR. Ask on the [LLVM forums](https://discourse.llvm.org) if you want to dig in.

### Language-Specific Optimizations

One common complaint about compiler infrastructure is that it doesn't solve every problem in one system. A specific complaint people have about LLVM is that it's perceived as incapable of high-level, language-specific optimization: the story goes that LLVM "loses too much information" once everything is lowered to a single fixed instruction set.

MLIR's answer to this is structurally different, and it's really what the later chapters of this tutorial are about: instead of bolting language-specific passes onto a fixed IR, MLIR lets you define **your own operations, types, and passes** — exactly as [Chapter 10](chapter-10.md) does for Kaleidoscope's variables.

A `kaleidoscope.var` operation isn't just an allocation wearing a different name. It preserves the source variable's name, location, and argument number until lowering has enough information to create both its storage and its debug declaration. *You* decide how long that information survives before converting it away — and while it's there, you can write analysis and rewrite patterns against it.

That said, MLIR isn't immune to the same tradeoff once you do lower. The `arith` and `llvm` dialects use structural type equivalence just like LLVM IR does, so two high-level types that both lower to `f64` or to `!llvm.struct<(i32)>` become indistinguishable once you're there (aside from debug info). The difference is that MLIR lets you choose, dialect by dialect, how much structure to keep before paying that cost. And just like with LLVM, if you hit a wall, the [MLIR Discourse](https://discourse.llvm.org/c/mlir/31) is a good place to ask.

Worst case, you can always treat any dialect as a "dumb code generator" and implement the high-level optimizations you want in your frontend, on the language-specific AST. More commonly, you preserve the information in MLIR and write transformations at the level where they make sense — just as [Chapter 4](chapter-04.md) used canonicalization and CSE before lowering further.

## Tips and Tricks

There are some useful techniques that aren't obvious at first glance but come up again and again once you've worked with MLIR for a while. Here are a few worth knowing.

### Implementing Portable `offsetof`/`sizeof`

If you're trying to keep the code your compiler generates target-independent, you'll often need to know the size of a type or the offset of a field in a structure. For example, you might need to pass a type's size into a memory allocation function.

The problem: this varies widely across targets. The width of a pointer alone is target-specific.

If you're already in the `llvm` dialect, the same [clever trick using `getelementptr`](http://nondot.org/sabre/LLVMNotes/SizeOf-OffsetOf-VariableSizedStructs.txt) that works in LLVM IR works unchanged, since `llvm.getelementptr` is the same operation with different syntax.

But MLIR also gives you a more direct, dialect-independent answer that avoids the GEP trick entirely: `mlir::DataLayout` exposes `getTypeSize` and `getTypeSizeInBits` for any type, resolved against whichever target data layout you're compiling for — the same data layout Chapter 4's JIT already retrieves from the `ExecutionEngine`/`KaleidoscopeJIT` before translating to LLVM IR.

### Garbage-Collected Stack Frames

Some languages want to explicitly manage their stack frames, often so they can be garbage collected or to make closures easier to implement. There are usually better ways to implement these features than explicit stack frames, but if you need them, the `llvm` dialect carries the same support LLVM does — via the same `garbageCollector` attribute mentioned earlier. It requires your frontend to convert code into [Continuation Passing Style](http://en.wikipedia.org/wiki/Continuation-passing_style) and use tail calls (which the `llvm` dialect also supports).


And that's the end of the tutorial. You've gone from a bare lexer to a full compiler with a custom dialect and real debug info. Whatever you build next, have fun with it.