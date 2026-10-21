"""RULA (Rapid Upper Limb Assessment) scoring. Pure Python, no Omniverse imports.

Tables A, B and C were transcribed cell by cell from the Ergonomics Plus "RULA Employee Assessment
Worksheet" (based on McAtamney & Corlett, Applied Ergonomics 1993, 24(2), 91-99) and checked against the
rendered worksheet. Angle conventions: flexion is positive, extension is negative, degrees.

Known differences between published RULA material (all exposed as parameters):
  * "static" posture: held more than 1 minute in most sources, more than 10 minutes in one worksheet.
  * Load thresholds follow the worksheet: under 2 kg = 0, 2-10 kg intermittent = 1, 2-10 kg static or
    repeated = 2, over 10 kg or shock = 3.
"""
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------- lookup tables
# Table A: TABLE_A[upper_arm 1-6][lower_arm 1-3] is 8 values ordered
# (wrist 1 twist 1, wrist 1 twist 2, wrist 2 twist 1, ..., wrist 4 twist 2)
TABLE_A = {
    1: {1: (1, 2, 2, 2, 2, 3, 3, 3), 2: (2, 2, 2, 2, 3, 3, 3, 3), 3: (2, 3, 3, 3, 3, 3, 4, 4)},
    2: {1: (2, 3, 3, 3, 3, 4, 4, 4), 2: (3, 3, 3, 3, 3, 4, 4, 4), 3: (3, 4, 4, 4, 4, 4, 5, 5)},
    3: {1: (3, 3, 4, 4, 4, 4, 5, 5), 2: (3, 4, 4, 4, 4, 4, 5, 5), 3: (4, 4, 4, 4, 4, 5, 5, 5)},
    4: {1: (4, 4, 4, 4, 4, 5, 5, 5), 2: (4, 4, 4, 4, 4, 5, 5, 5), 3: (4, 4, 4, 5, 5, 5, 6, 6)},
    5: {1: (5, 5, 5, 5, 5, 6, 6, 7), 2: (5, 6, 6, 6, 6, 7, 7, 7), 3: (6, 6, 6, 7, 7, 7, 7, 8)},
    6: {1: (7, 7, 7, 7, 7, 8, 8, 9), 2: (8, 8, 8, 8, 8, 9, 9, 9), 3: (9, 9, 9, 9, 9, 9, 9, 9)},
}

# Table B: TABLE_B[neck 1-6] is 12 values ordered (trunk 1 legs 1, trunk 1 legs 2, trunk 2 legs 1, ...)
TABLE_B = {
    1: (1, 3, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7),
    2: (2, 3, 2, 3, 4, 5, 5, 5, 6, 7, 7, 7),
    3: (3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7),
    4: (5, 5, 5, 6, 6, 7, 7, 7, 7, 7, 8, 8),
    5: (7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8),
    6: (8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9),
}

# Table C: TABLE_C[wrist & arm score 1-8 (8 = "8+")][neck, trunk & leg score 1-7 (7 = "7+")]
TABLE_C = {
    1: (1, 2, 3, 3, 4, 5, 5),
    2: (2, 2, 3, 4, 4, 5, 5),
    3: (3, 3, 3, 4, 4, 5, 6),
    4: (3, 3, 3, 4, 5, 6, 6),
    5: (4, 4, 4, 5, 6, 7, 7),
    6: (4, 4, 5, 6, 6, 7, 7),
    7: (5, 5, 6, 6, 7, 7, 7),
    8: (5, 5, 6, 7, 7, 7, 7),
}


def _clamp(value, lo, hi):
    return max(lo, min(hi, int(value)))


# ---------------------------------------------------------------- body part scores (steps 1-4, 9-11)
def score_upper_arm(angle_deg, shoulder_raised=False, abducted=False, supported=False):
    """Upper arm angle from the trunk: + forward (flexion), - backward (extension)."""
    if angle_deg < -20:
        base = 2
    elif angle_deg <= 20:
        base = 1
    elif angle_deg <= 45:
        base = 2
    elif angle_deg <= 90:
        base = 3
    else:
        base = 4
    return _clamp(base + int(shoulder_raised) + int(abducted) - int(supported), 1, 6)


def score_lower_arm(elbow_flexion_deg, across_midline_or_out_to_side=False):
    """Elbow angle: 60-100 degrees is the good range."""
    base = 1 if 60 <= elbow_flexion_deg <= 100 else 2
    return _clamp(base + int(across_midline_or_out_to_side), 1, 3)


def score_wrist(angle_deg, deviated=False, neutral_tol_deg=3.0):
    """Wrist flexion/extension: neutral = 1, up to 15 degrees = 2, more = 3; +1 if bent sideways."""
    a = abs(angle_deg)
    base = 1 if a <= neutral_tol_deg else (2 if a <= 15 else 3)
    return _clamp(base + int(deviated), 1, 4)


def score_wrist_twist(near_end_of_range=False):
    return 2 if near_end_of_range else 1


def score_neck(flexion_deg, twisted=False, side_bent=False):
    """Neck: 0-10 = 1, 10-20 = 2, over 20 = 3, extension = 4; +1 twisted, +1 side bending."""
    if flexion_deg < 0:
        base = 4
    elif flexion_deg <= 10:
        base = 1
    elif flexion_deg <= 20:
        base = 2
    else:
        base = 3
    return _clamp(base + int(twisted) + int(side_bent), 1, 6)


def score_trunk(flexion_deg, twisted=False, side_bent=False, upright_tol_deg=3.0):
    """Trunk: upright = 1, 0-20 = 2, 20-60 = 3, over 60 = 4; +1 twisted, +1 side bending."""
    if flexion_deg <= upright_tol_deg:
        base = 1
    elif flexion_deg <= 20:
        base = 2
    elif flexion_deg <= 60:
        base = 3
    else:
        base = 4
    return _clamp(base + int(twisted) + int(side_bent), 1, 6)


def score_legs(supported_and_balanced=True):
    return 1 if supported_and_balanced else 2


# ---------------------------------------------------------------- muscle use and force (steps 6, 7, 13, 14)
def muscle_use_score(static_seconds=0.0, repeats_per_minute=0.0, static_threshold_s=60.0):
    """+1 if the posture is mainly static (held longer than the threshold) or repeated 4+ times a minute."""
    return 1 if (static_seconds > static_threshold_s or repeats_per_minute >= 4) else 0


def calculate_force_score(mass_kg, is_repeated=False, is_static=False, is_shock=False):
    """Load / force score. Under 2 kg = 0; 2-10 kg intermittent = 1; 2-10 kg static or repeated = 2;
    over 10 kg, or a shock / rapid build-up of force = 3."""
    if is_shock:
        return 3
    if mass_kg is None or mass_kg < 2.0:
        return 0
    if mass_kg <= 10.0:
        return 2 if (is_repeated or is_static) else 1
    return 3


# ---------------------------------------------------------------- tables and final score
def table_a(upper_arm, lower_arm, wrist, wrist_twist):
    row = TABLE_A[_clamp(upper_arm, 1, 6)][_clamp(lower_arm, 1, 3)]
    return row[(_clamp(wrist, 1, 4) - 1) * 2 + (_clamp(wrist_twist, 1, 2) - 1)]


def table_b(neck, trunk, legs):
    return TABLE_B[_clamp(neck, 1, 6)][(_clamp(trunk, 1, 6) - 1) * 2 + (_clamp(legs, 1, 2) - 1)]


def table_c(wrist_arm_score, neck_trunk_leg_score):
    return TABLE_C[_clamp(wrist_arm_score, 1, 8)][_clamp(neck_trunk_leg_score, 1, 7) - 1]


def action_level(score):
    if score <= 2:
        return "Acceptable if not maintained or repeated for long periods"
    if score <= 4:
        return "Further investigation needed; changes may be required"
    if score <= 6:
        return "Further investigation and changes required soon"
    return "Investigation and changes required immediately"


@dataclass
class RulaResult:
    upper_arm: int
    lower_arm: int
    wrist: int
    wrist_twist: int
    neck: int
    trunk: int
    legs: int
    posture_a: int          # Table A
    posture_b: int          # Table B
    muscle_use: int
    force: int
    wrist_arm_score: int    # posture A + muscle use + force
    neck_trunk_leg_score: int
    grand_score: int        # Table C, 1-7
    action: str


def rula_score(upper_arm, lower_arm, wrist, wrist_twist, neck, trunk, legs,
               muscle_use=0, force=0):
    """Combine the body part scores (already adjusted) into the final RULA score.
    Muscle use and force are added to BOTH posture scores, then Table C gives the grand score."""
    a = table_a(upper_arm, lower_arm, wrist, wrist_twist)
    b = table_b(neck, trunk, legs)
    wa = a + muscle_use + force
    ntl = b + muscle_use + force
    grand = table_c(wa, ntl)
    return RulaResult(upper_arm, lower_arm, wrist, wrist_twist, neck, trunk, legs, a, b, muscle_use, force,
                      wa, ntl, grand, action_level(grand))


def rula_from_angles(upper_arm_deg, elbow_deg, wrist_deg, neck_deg, trunk_deg, *,
                     mass_kg=0.0, is_repeated=False, is_static=False, is_shock=False,
                     static_seconds=0.0, repeats_per_minute=0.0, static_threshold_s=60.0,
                     shoulder_raised=False, upper_arm_abducted=False, arm_supported=False,
                     arm_across_midline=False, wrist_deviated=False, wrist_twist_near_end=False,
                     neck_twisted=False, neck_side_bent=False, trunk_twisted=False, trunk_side_bent=False,
                     legs_supported=True) -> RulaResult:
    """One call from joint angles (degrees) plus the adjustment flags and the load."""
    return rula_score(
        score_upper_arm(upper_arm_deg, shoulder_raised, upper_arm_abducted, arm_supported),
        score_lower_arm(elbow_deg, arm_across_midline),
        score_wrist(wrist_deg, wrist_deviated),
        score_wrist_twist(wrist_twist_near_end),
        score_neck(neck_deg, neck_twisted, neck_side_bent),
        score_trunk(trunk_deg, trunk_twisted, trunk_side_bent),
        score_legs(legs_supported),
        muscle_use=muscle_use_score(static_seconds, repeats_per_minute, static_threshold_s) or int(is_static or is_repeated),
        force=calculate_force_score(mass_kg, is_repeated, is_static, is_shock),
    )
