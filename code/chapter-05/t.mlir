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
