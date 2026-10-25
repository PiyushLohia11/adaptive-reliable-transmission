"""Root convenience entry point to run full simulation experiments.

Delegates directly to scripts/run_experiments.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add scripts directory to path
SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_experiments import main

if __name__ == "__main__":
    main()
