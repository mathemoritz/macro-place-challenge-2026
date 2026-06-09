"""Placeholder script — documents how to populate ``submissions/lkh/seeds/``.

The placer is seeded from an analytical placement (e.g. RePlAce or
DREAMPlace). Neither tool ships in this repo, so seeds are produced
externally and dropped into ``seeds/<bench>.plc``.

Recommended path:

1. Install DREAMPlace (pip-installable, GPU-accelerated)::

       pip install dreamplace

   Or build RePlAce from source (https://github.com/The-OpenROAD-Project/RePlAce).

2. For each benchmark in the IBM suite (ibm01..ibm04, ibm06..ibm18), run
   the placer against ``external/MacroPlacement/Testcases/ICCAD04/<bench>/
   netlist.pb.txt`` + ``initial.plc``. The output should be a placement of
   the macros (positions only; net topology unchanged).

3. Convert the placement to the TILOS ``.plc`` format. This is the same
   format ``initial.plc`` uses — the easiest path is to write the new
   positions into a ``PlacementCost`` object loaded from the original
   netlist and call ``plc.save_placement(out_path, ...)``.

4. Save as ``submissions/lkh/seeds/<bench>.plc``. ``seed_loader`` will
   pick them up automatically when the placer runs with
   ``seed_source="replace"``.

This script is intentionally a placeholder — running it without an
externally-installed analytical placer prints the TODO and exits. The
loader contract is the source of truth; this file documents the producer.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SEEDS_DIR = _HERE / "seeds"

IBM_BENCHMARKS = [
    "ibm01", "ibm02", "ibm03", "ibm04",
    "ibm06", "ibm07", "ibm08", "ibm09",
    "ibm10", "ibm11", "ibm12", "ibm13",
    "ibm14", "ibm15", "ibm16", "ibm17", "ibm18",
]


def main() -> int:
    _SEEDS_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(p.stem for p in _SEEDS_DIR.glob("*.plc"))
    missing = [b for b in IBM_BENCHMARKS if b not in existing]

    print(f"seeds directory: {_SEEDS_DIR}")
    print(f"existing: {existing or '(none)'}")
    print(f"missing:  {missing or '(none)'}")

    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
