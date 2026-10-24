import os

import pytest

from frt_core.people.tasks import Task

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = os.path.join(ROOT, "docs", "people", "practice", "battery_box_scene.toml").replace("\\", "/")


def _task(tmp_path, steps):
    body = [f'[task]\nname = "t"\nscene = "{SCENE}"\nworker = "W50M"\nstart = [-300, -60, 60]\n']
    for s in steps:
        body.append("[[step]]\n" + "".join(f'{k} = {v!r}\n'.replace("'", '"') for k, v in s.items()))
    p = tmp_path / "t.toml"
    p.write_text("\n".join(body))
    return Task.load(str(p))


def test_tool_kept_in_hand(tmp_path):
    t = _task(tmp_path, [{"do": "get_tool", "tool": "ratchet"},
                         {"do": "hand_start", "what": "nut", "on": "battery_box.studs", "count": 2},
                         {"do": "fasten", "tool": "ratchet", "on": "battery_box.studs"},
                         {"do": "put_back", "tool": "ratchet"}]).compile(log=lambda *a: None)
    labels = [f.label for f in t.frames]
    assert any(lab.endswith("swap hands") for lab in labels)      # passed to the left hand for the nuts
    assert not any(lab.startswith("03_fasten:get tool") for lab in labels)
    a = next(i for i, lab in enumerate(labels) if lab.startswith("01_get_tool:hold"))
    b = next(i for i, lab in enumerate(labels) if lab.startswith("04_put_back:release"))
    assert all("part:ratchet" in f.tools for f in t.frames[a:b])  # in hand all the way between
    assert t.holding is None


def test_tool_in_hand_errors(tmp_path):
    with pytest.raises(ValueError, match="does not hold"):
        _task(tmp_path, [{"do": "put_back", "tool": "ratchet"}]).compile(log=lambda *a: None)
    with pytest.raises(ValueError, match="holds ratchet"):
        _task(tmp_path, [{"do": "get_tool", "tool": "ratchet"},
                         {"do": "fasten", "tool": "screwdriver", "on": "cover.holes"}]).compile(log=lambda *a: None)
