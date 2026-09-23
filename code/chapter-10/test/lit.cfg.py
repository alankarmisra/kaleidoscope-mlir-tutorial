import os
import lit.formats

config.name = "Kaleidoscope Chapter 10"
config.test_format = lit.formats.ShTest(False)
config.suffixes = [".ks"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.toy_build_dir, "test")

config.substitutions.append(("%toy", os.path.join(config.toy_build_dir, "toy")))
config.substitutions.append(("%FileCheck", os.path.join(config.llvm_tools_dir, "FileCheck")))
