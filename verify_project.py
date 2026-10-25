"""Root convenience entry point to run project verification tests.

Delegates directly to scripts/verify_project.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add scripts directory to path
SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from verify_project import main

if __name__ == "__main__":
    main()
