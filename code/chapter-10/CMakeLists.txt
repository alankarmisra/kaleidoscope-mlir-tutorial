# CMakeLists.txt

cmake_minimum_required(VERSION 3.20)

project(kaleidoscope-chapter-10 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(TableGen)
include(AddMLIR)
include_directories(${MLIR_INCLUDE_DIRS})

# Every mlir_tablegen invocation below reads this TableGen source file.
set(LLVM_TARGET_DEFINITIONS KaleidoscopeOps.td)

# Generate the KaleidoscopeDialect class declaration and definition from the
# `def Kaleidoscope_Dialect : Dialect` record.
mlir_tablegen(KaleidoscopeDialect.h.inc -gen-dialect-decls
  -dialect=kaleidoscope)
mlir_tablegen(KaleidoscopeDialect.cpp.inc -gen-dialect-defs
  -dialect=kaleidoscope)

# Generate C++ declarations and definitions for our TypeDef records.
mlir_tablegen(KaleidoscopeTypes.h.inc -gen-typedef-decls
  -typedefs-dialect=kaleidoscope)
mlir_tablegen(KaleidoscopeTypes.cpp.inc -gen-typedef-defs
  -typedefs-dialect=kaleidoscope)

# Generate C++ declarations and definitions for our Op records.
mlir_tablegen(KaleidoscopeOps.h.inc -gen-op-decls)
mlir_tablegen(KaleidoscopeOps.cpp.inc -gen-op-defs)

# Give the generated fragments one build target so the executable can depend
# on all of them before compiling files that #include those fragments.
add_custom_target(KaleidoscopeOpsIncGen DEPENDS
  KaleidoscopeDialect.h.inc
  KaleidoscopeDialect.cpp.inc
  KaleidoscopeTypes.h.inc
  KaleidoscopeTypes.cpp.inc
  KaleidoscopeOps.h.inc
  KaleidoscopeOps.cpp.inc
)

add_executable(toy
  toy.cpp
  KaleidoscopeDialect.cpp
  KaleidoscopeDebugInfo.cpp
)
add_dependencies(toy KaleidoscopeOpsIncGen)

# Make symbols in the executable available to the JIT for runtime lookup.
set_target_properties(toy PROPERTIES ENABLE_EXPORTS ON)

target_include_directories(toy PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)

target_compile_definitions(toy PRIVATE ${LLVM_DEFINITIONS})

if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(toy PRIVATE -fno-rtti)
endif()

llvm_map_components_to_libnames(LLVM_LIBS
  Core
  CodeGen
  OrcJIT
  Support
  Target
  ${LLVM_TARGETS_TO_BUILD}
)

target_link_libraries(toy PRIVATE
  MLIRArithToLLVM
  MLIRArithDialect
  MLIRBuiltinToLLVMIRTranslation
  MLIRControlFlowDialect
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRLLVMIRTransforms
  MLIRLLVMToLLVMIRTranslation
  MLIRMemRefDialect
  MLIRMemRefToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRSCFDialect
  MLIRSCFToControlFlow
  MLIRTargetLLVMIRExport
  MLIRTransforms
  ${LLVM_LIBS}
)
