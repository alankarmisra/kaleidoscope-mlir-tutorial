declare double @foo()
declare double @bar()

define double @baz(double %x) {
entry:
  %ifcond = fcmp one double %x, 0.000000e+00
  br i1 %ifcond, label %then, label %else

then:
  %calltmp = call double @foo()
  br label %ifcont

else:
  %calltmp1 = call double @bar()
  br label %ifcont

ifcont:
  %iftmp = phi double [ %calltmp, %then ], [ %calltmp1, %else ]
  ret double %iftmp
}
