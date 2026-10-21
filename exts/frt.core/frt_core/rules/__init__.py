"""Rule packs: limits as data. Packs are TOML files; later packs override earlier ones key by key.

    rules = load()                                   # the toolkit defaults
    rules = load(["my_plant.toml"])                  # defaults + overrides
    rules.get("layout.main_aisle_min_cm")            # 360
"""
import copy
import os

try:
    import tomllib
except ImportError:        # Python < 3.11
    import tomli as tomllib

DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "default.toml")


def _merge(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _merge(a[k], v)
        else:
            a[k] = copy.deepcopy(v)
    return a


class Rules:
    def __init__(self, data, sources):
        self.data, self.sources = data, sources

    def get(self, key, default=None):
        node = self.data
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def __getitem__(self, key):
        v = self.get(key)
        if v is None:
            raise KeyError(key)
        return v


def load(packs=(), include_default=True):
    data, sources = {}, []
    for p in ([DEFAULT] if include_default else []) + list(packs):
        with open(p, "rb") as fh:
            _merge(data, tomllib.load(fh))
        sources.append(p)
    return Rules(data, sources)
