# RUN: %toy --dump-mlir < %s 2>&1 | %FileCheck %s
# RUN: %toy --dump-llvm-ir < %s 2>&1 | %FileCheck %s --check-prefix=LLVM

def arithmetic(x y) (x + y) * y;
arithmetic(2, 3);

def folded() 2 + 3;

def binary % 40 (x y) x * y;
def custom(x y) x % y;
custom(6, 7);

# CHECK: %{{.*}} = kaleidoscope.binary "+"
# CHECK: %{{.*}} = kaleidoscope.binary "*"
# CHECK: Evaluated to 15.000000
# CHECK: arith.constant 5.000000e+00
# CHECK: %{{.*}} = kaleidoscope.binary "%"
# CHECK: Evaluated to 42.000000
# LLVM: define double @arithmetic
