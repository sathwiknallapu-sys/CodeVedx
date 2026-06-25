"""
conftest.py
-------------
Pytest automatically loads this file before running tests.
It adds the src/ directory to sys.path so test files can do plain
imports like `from validators import ...` without needing the
project installed as a package.
"""

import os
import sys

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)
