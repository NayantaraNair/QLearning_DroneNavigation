from __future__ import annotations

"""Entry point for the drone Q-learning GUI application."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    package_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(package_dir.parent))
    from mlda.gui import launch_gui
else:
    from .gui import launch_gui


def main() -> None:
    launch_gui()


if __name__ == "__main__":
    main()
