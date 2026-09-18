"""Enable ``python -m saryolo ...`` as the primary entry point.

This matters for Colab: the repository can be used straight from a clone without
pip-installing the package, which keeps the notebooks simple and avoids a
stale installed copy shadowing the working tree.
"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
