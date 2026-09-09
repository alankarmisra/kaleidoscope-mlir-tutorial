# My First Language Frontend with MLIR Tutorial

**Requirements:** This tutorial assumes you know C++, but no previous compiler experience is necessary.

Welcome to the "My First Language Frontend with MLIR" tutorial. Here we run through the implementation of a simple language, showing how fun and easy it can be. This tutorial will get you up and running fast and show a concrete example of something that uses MLIR to generate code.

This tutorial introduces the simple "Kaleidoscope" language, building it iteratively over the course of several chapters, showing how it is built over time. This lets us cover a range of language design and MLIR-specific ideas, showing and explaining the code for it all along the way, and reduces the overwhelming amount of details up front. We strongly encourage that you *work with this code* - make a copy and hack it up and experiment.

**Warning**: In order to focus on teaching compiler techniques and MLIR specifically, this tutorial does *not* show best practices in software engineering principles. For example, the code uses global variables pervasively, doesn't use [visitors](http://en.wikipedia.org/wiki/Visitor_pattern), etc... but instead keeps things simple and focuses on the topics at hand.

This tutorial is structured into chapters covering individual topics, allowing you to skip ahead as you wish:

- [Chapter #1: Kaleidoscope language and Lexer](chapter-01.md) - This shows where we are going and the basic functionality that we want to build. A lexer is also the first part of building a parser for a language, and we use a simple C++ lexer which is easy to understand.
- [Chapter #2: Implementing a Parser and AST](chapter-02.md) - With the lexer in place, we can talk about parsing techniques and basic AST construction. This tutorial describes recursive descent parsing and operator precedence parsing.
- [Chapter #3: Code generation to MLIR](chapter-03.md) - with the AST ready, we show how easy it is to generate MLIR, and show a simple way to incorporate MLIR into your project.
- [Chapter #4: Adding JIT and Optimizer Support](chapter-04.md) - One great thing about MLIR is its integration with LLVM's JIT infrastructure, so we'll dive right into it and show you how few lines it takes to add JIT support. Later chapters show how to generate .o files.
- [Chapter #5: Extending the Language: Control Flow](chapter-05.md) - With the basic language up and running, we show how to extend it with control flow operations ('if' statement and a 'for' loop). This gives us a chance to talk about SSA construction and control flow.
- [Chapter #6: Extending the Language: User-defined Operators](chapter-06.md) - This chapter extends the language to let users define arbitrary unary and binary operators - with assignable precedence! This allows us to build a significant piece of the "language" as library routines.
- [Chapter #7: Extending the Language: Mutable Variables](chapter-07.md) - This chapter talks about adding user-defined local variables along with an assignment operator. This shows how easy it is to handle mutable variables in MLIR: MLIR does *not* require your front-end to construct SSA form for mutable variables in order to use it!
- [Chapter #8: Compiling to Object Files](chapter-08.md) - This chapter explains how to take MLIR and compile it down to object files, like a static compiler does.
- [Chapter #9: Debug Information](chapter-09.md) - A real language needs to support debuggers, so we add debug information that allows setting breakpoints in Kaleidoscope functions, print out argument variables, and call functions!
- [Chapter #10: Conclusion and other tidbits](chapter-10.md) - This chapter wraps up the series by discussing ways to extend the language and includes pointers to info on "special topics" like adding garbage collection support, exceptions, debugging, support for "spaghetti stacks", etc.

By the end of the tutorial, we'll have built up a nice little compiler for a non-trivial language including a hand-written lexer, parser, AST, as well as code generation support - both static and JIT! The breadth of this is a great testament to the strengths of MLIR and LLVM, and shows why they provide such a powerful foundation for language designers and others who need high performance code generation.
