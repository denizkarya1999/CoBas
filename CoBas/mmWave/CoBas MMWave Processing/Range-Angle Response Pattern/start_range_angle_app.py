#!/usr/bin/env python3
"""Start the live range-angle GUI from any working directory."""

from __future__ import annotations

import sys
from pathlib import Path


APPLICATION_DIRECTORY = Path(__file__).resolve().parent
application_path = str(APPLICATION_DIRECTORY)
if application_path not in sys.path:
    sys.path.insert(0, application_path)

from Interface.range_angle_interface import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
