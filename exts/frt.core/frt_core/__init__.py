"""Factory Review Toolkit core: everything that does not need Omniverse Kit (pure Python + OpenUSD).

Conventions used across the toolkit (scenes it writes): Y up, centimetres (metersPerUnit 0.01), 30 fps. Assets in
other conventions are wrapped, never edited (units.py). Every check returns Findings (findings.py).
"""
__version__ = "0.1.0"

UP_AXIS = "Y"
METERS_PER_UNIT = 0.01
FPS = 30.0
