cmake_minimum_required(VERSION 3.20)

project(kaleidoscope-chapter-03 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

find_package(MLIR REQUIRED CONFIG)

add_executable(toy toy.cpp)

if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(toy PRIVATE -fno-rtti)
endif()

target_include_directories(toy PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)

target_compile_definitions(toy PRIVATE ${LLVM_DEFINITIONS})

target_link_libraries(toy PRIVATE
  MLIRArithDialect
  MLIRFuncDialect
)
