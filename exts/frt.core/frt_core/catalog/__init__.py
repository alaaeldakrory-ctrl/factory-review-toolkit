"""Equipment catalog. `CATALOG` maps item keys to Item objects (base.py); importing this package registers all
items (items.py)."""
from .base import CATALOG, Item, Param, place, register   # noqa: F401
from . import items   # noqa: F401  (registers the items)
