"""Equipment catalog. `CATALOG` maps item keys to Item objects (base.py); importing this package registers all
items (items.py, items_lifting.py)."""
from .base import CATALOG, Item, Param, place, register   # noqa: F401
from . import items, items_lifting, items_station, items_crane   # noqa: F401  (registers the items; later wins)
