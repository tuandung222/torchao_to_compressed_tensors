#!/usr/bin/env python3
"""
Backward-compatibility shim for torchao_to_compressed_tensors_adapter.
Delegates to the modular src.torchao_to_compressed_tensors package.
"""

import sys
from pathlib import Path

# Add src to python path for direct script execution
_src_dir = Path(__file__).resolve().parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from torchao_to_compressed_tensors import *
from torchao_to_compressed_tensors.adapter import main

if __name__ == "__main__":
    main()
