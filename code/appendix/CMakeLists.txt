cmake_minimum_required(VERSION 3.20)

project(kaleidoscope-chapter-04 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

find_package(MLIR REQUIRED CONFIG)

add_executable(toy toy.cpp)

target_include_directories(toy PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)

target_compile_definitions(toy PRIVATE ${LLVM_DEFINITIONS})

llvm_map_components_to_libnames(LLVM_LIBS
  Core
  OrcJIT
  native
)

target_link_libraries(toy PRIVATE
  MLIRArithToLLVM
  MLIRArithDialect
  MLIRBuiltinToLLVMIRTranslation
  MLIRFuncToLLVM
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRLLVMToLLVMIRTranslation
  MLIRReconcileUnrealizedCasts
  MLIRTargetLLVMIRExport
  MLIRTransforms
  ${LLVM_LIBS}
)
