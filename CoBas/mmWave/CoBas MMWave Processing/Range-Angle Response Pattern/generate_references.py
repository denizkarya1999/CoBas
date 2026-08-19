#!/usr/bin/env python3
"""Generate missing reference images for all saved battery frame folders."""

from __future__ import annotations

import sys
from pathlib import Path


APPLICATION_DIRECTORY = Path(__file__).resolve().parent
application_path = str(APPLICATION_DIRECTORY)
if application_path not in sys.path:
    sys.path.insert(0, application_path)

from Logic.reference_frame_generator import (  # noqa: E402
    generate_references_for_saved_batteries,
)


def main() -> int:
    results = generate_references_for_saved_batteries()
    if not results:
        print("No battery frame folders were found.")
        return 0

    for result in results:
        action = "Created" if result.created else "Kept existing"
        source = (
            f" from {result.selected_frame.name}"
            if result.selected_frame is not None
            else ""
        )
        print(f"{action}: {result.reference_image}{source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
