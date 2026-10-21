"""Scene tagging: what each object is. Tags are plain USD attributes in the `frt:` namespace on the object's root
prim, so they survive copying, referencing and other tools:

    frt:role     worker / forklift / conveyor / table / lift_table / flow_rack / pallet / ... (ROLES)
    frt:id       stable id (from the drawing element or the recipe), used by revision sync
    frt:source   where it came from: "dxf:<handle>", "pdf:<index>", "recipe", "manual"
    frt:catalog  catalog generator or asset key; frt:params  JSON of its parameters

auto_tag() guesses roles from prim names for scenes built by hand.
"""
import json
import re

from pxr import Sdf, Usd

ROLES = (
    "worker", "forklift", "pallet_jack", "conveyor", "table", "lift_table", "flow_rack", "bin_rack", "pallet_rack",
    "pallet", "crate", "crane_bridge", "crane_jib", "machine", "robot", "column", "wall", "door", "dock", "aisle",
    "walkway", "zone", "keepout", "exit", "fire_equipment", "floor", "building", "camera", "prop",
)
NAME_PATTERNS = [   # (regex on the prim name, role)
    (r"forklift", "forklift"), (r"palletjack|pallet_jack|handpallet", "pallet_jack"), (r"conveyor", "conveyor"),
    (r"lift_?table|scissor_?table", "lift_table"), (r"flow_?rack|carton_?flow", "flow_rack"),
    (r"bin_?rack|louver", "bin_rack"), (r"rack", "pallet_rack"), (r"pallet", "pallet"), (r"crate", "crate"),
    (r"bridge_?crane|overhead_?crane|gantry", "crane_bridge"), (r"jib", "crane_jib"),
    (r"workbench|work_?table|table|bench", "table"), (r"worker|character|human|operator|uniform_[mf]", "worker"),
    (r"column|pillar", "column"), (r"extinguisher|fire_?box|hydrant", "fire_equipment"), (r"dock", "dock"),
    (r"door", "door"), (r"exit", "exit"), (r"robot|ur\d|kuka|fanuc|abb_irb", "robot"),
]


def tag(prim, role, id=None, source=None, catalog=None, params=None):
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}")
    prim.CreateAttribute("frt:role", Sdf.ValueTypeNames.Token, custom=True).Set(role)
    if id is not None:
        prim.CreateAttribute("frt:id", Sdf.ValueTypeNames.String, custom=True).Set(str(id))
    if source is not None:
        prim.CreateAttribute("frt:source", Sdf.ValueTypeNames.String, custom=True).Set(str(source))
    if catalog is not None:
        prim.CreateAttribute("frt:catalog", Sdf.ValueTypeNames.String, custom=True).Set(str(catalog))
    if params is not None:
        prim.CreateAttribute("frt:params", Sdf.ValueTypeNames.String, custom=True).Set(json.dumps(params, sort_keys=True))
    return prim


def _get(prim, name):
    a = prim.GetAttribute(name)
    return a.Get() if a and a.HasAuthoredValue() else None


def role_of(prim):
    return _get(prim, "frt:role")


def id_of(prim):
    return _get(prim, "frt:id")


def params_of(prim):
    p = _get(prim, "frt:params")
    return json.loads(p) if p else {}


def tagged(stage, role=None, roles=None):
    """Tagged prims (the roots carrying frt:role), optionally of one role or a set of roles. Does not descend into
    a tagged prim's children."""
    want = {role} if role else (set(roles) if roles else None)
    out = []
    it = iter(Usd.PrimRange(stage.GetPseudoRoot()))
    for p in it:
        r = role_of(p)
        if r:
            if want is None or r in want:
                out.append(p)
            it.PruneChildren()
    return out


def guess_role(name):
    n = name.lower()
    for pat, role in NAME_PATTERNS:
        if re.search(pat, n):
            return role
    return None


def auto_tag(stage, root="/World", overwrite=False):
    """Tag untagged prims by name (first match wins, children of a matched prim are skipped). Returns
    [(path, role)] of what was tagged."""
    done = []
    start = stage.GetPrimAtPath(root) or stage.GetPseudoRoot()
    it = iter(Usd.PrimRange(start))
    for p in it:
        if p == start:
            continue
        if role_of(p) and not overwrite:
            it.PruneChildren()
            continue
        if p.GetTypeName() in ("Material", "Shader", "NodeGraph", "GeomSubset"):
            it.PruneChildren()
            continue
        r = guess_role(p.GetName())
        if r:
            tag(p, r, source="auto")
            done.append((str(p.GetPath()), r))
            it.PruneChildren()
    return done
