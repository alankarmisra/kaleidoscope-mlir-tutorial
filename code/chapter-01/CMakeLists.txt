cmake_minimum_required(VERSION 3.20)

project(kaleidoscope-mlir-chapter-01 LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddLLVM)
include(AddMLIR)

include_directories(SYSTEM
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)

add_definitions(${LLVM_DEFINITIONS})

add_executable(kaleidoscope kaleidoscope.cpp)
llvm_update_compile_flags(kaleidoscope)

target_link_libraries(kaleidoscope PRIVATE
  MLIRArithDialect
  MLIRFuncDialect
)
