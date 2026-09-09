import os

import lit.formats

config.name = "kaleidoscope-mlir-chapter-01"
config.test_format = lit.formats.ShTest(True)
config.suffixes = [".test"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = config.test_source_root

chapter_dir = os.path.abspath(os.path.join(config.test_source_root, ".."))
config.substitutions.append(
    ("%kaleidoscope", os.path.join(chapter_dir, "build", "kaleidoscope"))
)

llvm_build = os.environ.get(
    "KALEIDOSCOPE_LLVM_BUILD",
    "/Users/alankar/Documents/opensource/llvm-project/build",
)
config.substitutions.append(
    ("%FileCheck", os.path.join(llvm_build, "bin", "FileCheck"))
)
