"""Ergonomic scan: joint names of common rigs are recognised, and posture angles turn into the right RULA bands."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "exts", "frt.core"))
from frt_core.people import ergo_scan as E   # noqa: E402

MIXAMO = ["mixamorig:Hips", "mixamorig:Spine2", "mixamorig:Neck", "mixamorig:Head", "mixamorig:LeftShoulder",
          "mixamorig:LeftArm", "mixamorig:LeftForeArm", "mixamorig:LeftHand", "mixamorig:RightArm",
          "mixamorig:RightForeArm", "mixamorig:RightHand", "mixamorig:LeftUpLeg", "mixamorig:LeftLeg",
          "mixamorig:LeftFoot", "mixamorig:RightUpLeg", "mixamorig:RightLeg", "mixamorig:RightFoot"]
CC = ["Hip", "Pelvis", "Spine01", "Spine02", "NeckTwist01", "Head", "L_Clavicle", "L_Upperarm", "L_Forearm", "L_Hand",
      "R_Upperarm", "R_Forearm", "R_Hand", "L_Thigh", "L_Calf", "L_Foot", "R_Thigh", "R_Calf", "R_Foot"]


def test_rigs_recognised():
    for rig in (MIXAMO, CC):
        m = E.map_joints(rig)
        assert all(r in m for r in E.REQUIRED), [r for r in E.REQUIRED if r not in m]
    assert MIXAMO[E.map_joints(MIXAMO)["upperarm_l"]] == "mixamorig:LeftArm"      # not the clavicle


def _body(trunk_deg=0.0, arm_up_deg=0.0):
    """Joint positions (Y-up cm, facing +Z) of a standing body with the trunk bent forward and the right arm raised."""
    a = np.radians(trunk_deg)
    up = np.array([0.0, np.cos(a), np.sin(a)])
    hip = np.array([0.0, 100.0, 0.0])
    ch = hip + up * 45
    J = {"pelvis": hip, "chest": ch, "neck": ch + up * 8, "head": ch + up * 20}
    for s, x in (("l", 1), ("r", -1)):
        sh = ch + np.array([18.0 * x, 0, 0])
        b = np.radians(arm_up_deg if s == "r" else 0.0)
        d = -up * np.cos(b) + np.array([0, 0, 1.0]) * np.sin(b)
        el = sh + d * 30
        J[f"upperarm_{s}"], J[f"forearm_{s}"] = sh, el
        J[f"hand_{s}"] = el + np.array([0, 0, 1.0]) * 26 + d * 2
        J[f"thigh_{s}"] = hip + np.array([9.0 * x, -5, 0])
        J[f"calf_{s}"] = J[f"thigh_{s}"] + np.array([0, -45.0, 0])
        J[f"foot_{s}"] = J[f"calf_{s}"] + np.array([0, -44.0, 0])
    return J


def test_bands():
    rest = {"trunk": 0.0, "neck": 0.0, "wrist_l": 0.0, "wrist_r": 0.0}
    upright = E.score(E.frame_angles(_body()), rest)
    bent = E.score(E.frame_angles(_body(trunk_deg=65, arm_up_deg=60)), rest)    # trunk past 60 deg: RULA 5+
    assert upright["score"] <= 3, upright
    assert bent["score"] >= 5, bent
    assert "trunk bent" in E.driver(bent) or "arm raised" in E.driver(bent)
