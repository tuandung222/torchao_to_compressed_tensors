#!/usr/bin/env python3
"""
Backward-compatibility shim for test_converter_parity.py.
Delegates to tests.test_parity.main().
"""

import sys
from pathlib import Path

# Add tests and src to sys.path
_repo_dir = Path(__file__).resolve().parent
if str(_repo_dir / "src") not in sys.path:
    sys.path.insert(0, str(_repo_dir / "src"))
if str(_repo_dir / "tests") not in sys.path:
    sys.path.insert(0, str(_repo_dir / "tests"))

from tests.test_parity import main

if __name__ == "__main__":
    main()
