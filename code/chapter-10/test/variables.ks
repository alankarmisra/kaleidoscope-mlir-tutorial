# RUN: %toy --dump-mlir < %s 2>&1 | %FileCheck %s
# RUN: %toy --dump-llvm-ir < %s 2>&1 | %FileCheck %s --check-prefix=LLVM

def test(x) var y = x in (y = y + 1) * y;
test(4);

# CHECK: kaleidoscope.var "x"
# CHECK: kaleidoscope.read
# CHECK: kaleidoscope.var "y"
# CHECK: kaleidoscope.assign
# CHECK: kaleidoscope.read
# CHECK: Evaluated to 25.000000
# LLVM: alloca double
# LLVM: #dbg_declare
# LLVM: !DILocalVariable(name: "x", arg: 1
# LLVM: !DILocalVariable(name: "y"
