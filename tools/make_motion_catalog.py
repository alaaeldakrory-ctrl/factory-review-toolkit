"""Writes exts/frt.core/frt_core/people/data/motions.toml from the compact table below (run once; afterwards the
TOML is the source of truth and can be edited by hand).

Row: id, name, hands, tool, object, postures, pattern, uas blocks, params, risks
  hands: one | two | either | dom+sup (dominant works, support hand holds / braces) | team
  postures: S standing, K kneeling, Q squatting, O overhead, T seated, L lying (creeper), P on step stand
  uas: MTM-UAS building blocks - GP get & place, PL place, HT handle tool, OP operate, MC motion cycle,
       FL fasten / loosen, BM body motion, VC visual control; parameters in brackets
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "exts", "frt.core", "frt_core", "people", "data", "motions.toml")

C = {}   # category -> rows

C["basic"] = [
    ("basic.reach_grasp.near", "Reach and grasp, near (<20 cm)", "either", "", "part", "S T", "reach_place", ["GP[d<20,w<1kg,easy]"], {}, []),
    ("basic.reach_grasp.mid", "Reach and grasp, mid (20-50 cm)", "either", "", "part", "S T", "reach_place", ["GP[d20-50,w<1kg,easy]"], {}, ["reach_zone"]),
    ("basic.reach_grasp.far", "Reach and grasp, far (50-80 cm)", "either", "", "part", "S", "reach_place", ["GP[d50-80,w<1kg,easy]"], {}, ["reach_zone", "trunk_flexion"]),
    ("basic.grasp.difficult", "Grasp jumbled / small part", "one", "", "small_part", "S T", "reach_place", ["GP[d20-50,w<1kg,difficult]"], {}, ["pinch_grip"]),
    ("basic.place.loose", "Place loosely", "either", "", "part", "S T", "reach_place", ["PL[loose]"], {}, []),
    ("basic.place.precise", "Place with alignment", "either", "", "part", "S T", "reach_place", ["PL[tight]"], {}, ["wrist_deviation"]),
    ("basic.place.two_hands", "Get and place with both hands", "two", "", "part", "S", "reach_place", ["GP[d20-50,w1-8kg,two_hands]"], {}, ["load"]),
    ("basic.regrasp", "Regrasp / change grip", "one", "", "part", "S T", "reach_place", ["GP[d<20,regrasp]"], {}, ["waste_regrasp"]),
    ("basic.transfer_hands", "Pass part to other hand", "two", "", "part", "S T", "reach_place", ["GP[d<20,transfer]"], {}, []),
    ("basic.release", "Release", "either", "", "part", "S T", "reach_place", ["PL[release]"], {}, []),
    ("basic.hold", "Hold part in position", "either", "", "part", "S K O", "operate", ["PL[hold]"], {"seconds": 3}, ["static_hold"]),
    ("basic.visual_check", "Look / visual check", "none", "", "point", "S K Q T", "inspect", ["VC"], {}, ["neck_flexion"]),
    ("basic.read_document", "Read work instruction / label", "one", "", "document", "S T", "inspect", ["VC", "GP[d20-50]"], {}, []),
    ("basic.press_button", "Press push button", "one", "", "button", "S T", "operate", ["OP[button]"], {}, []),
    ("basic.turn_knob", "Turn knob / selector", "one", "", "knob", "S T", "operate", ["OP[turn]"], {}, ["wrist_deviation"]),
    ("basic.operate_lever", "Operate lever", "one", "", "lever", "S T", "operate", ["OP[lever]"], {"force_n": 50}, ["force"]),
    ("basic.apply_pressure", "Apply pressure (press in)", "either", "", "part", "S K", "operate", ["OP[apply_pressure]"], {"force_n": 80}, ["force", "thumb"]),
    ("body.bend_arise", "Bend and arise", "none", "", "floor", "S", "body", ["BM[bend]"], {}, ["trunk_flexion", "repetition"]),
    ("body.kneel_one", "Kneel on one knee and arise", "none", "", "floor", "S K", "body", ["BM[kneel_one]"], {}, ["knee_contact"]),
    ("body.kneel_both", "Kneel on both knees and arise", "none", "", "floor", "S K", "body", ["BM[kneel_both]"], {}, ["knee_contact"]),
    ("body.squat", "Squat and arise", "none", "", "floor", "S Q", "body", ["BM[squat]"], {}, ["knee_flexion"]),
    ("body.sit_stand", "Sit down / stand up", "none", "", "seat", "S T", "body", ["BM[sit]"], {}, []),
    ("body.turn", "Turn body 90 deg", "none", "", "direction", "S", "body", ["BM[turn]"], {"angle_deg": 90}, ["twist_with_load"]),
    ("body.step", "Step (side / forward)", "none", "", "point", "S", "walk", ["BM[step]"], {"steps": 1}, []),
    ("body.reach_overhead", "Reach overhead", "either", "", "point", "S P", "reach_place", ["GP[d50-80,overhead]"], {}, ["shoulder_elevation"]),
    ("body.lie_creeper", "Get on / off creeper", "none", "creeper", "floor", "L", "body", ["BM[lie]"], {}, ["neck_extension"]),
]

C["handling"] = [
    ("handle.pick.bin_front", "Pick part from front bin", "one", "", "bin", "S", "reach_place", ["GP[d20-50,w<1kg]"], {}, ["reach_zone"]),
    ("handle.pick.bin_low", "Pick part from low bin", "one", "", "bin", "S Q", "reach_place", ["GP[d50-80,w<1kg]", "BM[bend]"], {}, ["trunk_flexion"]),
    ("handle.pick.bin_high", "Pick part from high bin", "one", "", "bin", "S", "reach_place", ["GP[d50-80,w<1kg]"], {}, ["shoulder_elevation"]),
    ("handle.pick.flow_lane", "Pick from flow rack lane", "one", "", "flow_lane", "S", "reach_place", ["GP[d20-50,w1-8kg]"], {}, ["reach_zone"]),
    ("handle.pick.tote", "Take tote from flow rack", "two", "", "tote", "S", "reach_place", ["GP[d20-50,w1-8kg,two_hands]"], {}, ["load"]),
    ("handle.return.tote", "Return empty tote to return lane", "two", "", "tote", "S", "reach_place", ["GP[d50-80,w<1kg,two_hands]"], {}, ["shoulder_elevation"]),
    ("handle.place.fixture", "Place part into fixture", "two", "", "fixture", "S", "reach_place", ["PL[tight]"], {}, []),
    ("handle.carry.small", "Carry small part", "one", "", "part", "S", "carry", ["GP", "BM[walk]"], {}, []),
    ("handle.carry.two_hand", "Carry load with both hands", "two", "", "part", "S", "carry", ["GP[w8-22kg]", "BM[walk]"], {"mass_kg": 10}, ["niosh", "load"]),
    ("handle.carry.long", "Carry long part (two people)", "team", "", "long_part", "S", "team", ["GP[team]", "BM[walk]"], {"mass_kg": 30}, ["niosh", "coordination"]),
    ("handle.lift.floor", "Lift from floor (squat lift)", "two", "", "part", "S Q", "reach_place", ["GP[w8-22kg]", "BM[squat]"], {"mass_kg": 12}, ["niosh", "trunk_flexion"]),
    ("handle.lift.pallet", "Lift from pallet", "two", "", "pallet", "S", "reach_place", ["GP[w8-22kg]", "BM[bend]"], {"mass_kg": 10}, ["niosh"]),
    ("handle.lift.team", "Team lift and place", "team", "", "part", "S", "team", ["GP[team]"], {"mass_kg": 40}, ["niosh", "coordination"]),
    ("handle.handoff.give", "Hand part to coworker", "either", "", "worker", "S", "team", ["GP[transfer]"], {}, ["coordination"]),
    ("handle.handoff.take", "Take part from coworker", "either", "", "worker", "S", "team", ["GP[transfer]"], {}, ["coordination"]),
    ("handle.cart.push", "Push cart", "two", "", "cart", "S", "carry", ["OP[push]", "BM[walk]"], {"force_n": 150}, ["push_pull"]),
    ("handle.cart.pull", "Pull cart", "two", "", "cart", "S", "carry", ["OP[pull]", "BM[walk]"], {"force_n": 150}, ["push_pull", "twist_with_load"]),
    ("handle.cart.load", "Load part onto cart", "two", "", "cart", "S", "reach_place", ["GP[w1-8kg]"], {}, []),
    ("handle.cart.unload", "Unload part from cart", "two", "", "cart", "S", "reach_place", ["GP[w1-8kg]"], {}, []),
    ("handle.lift_table.tow", "Tow lift table by handle", "two", "lift_table", "lift_table", "S", "carry", ["OP[pull]", "BM[walk]"], {"force_n": 200}, ["push_pull"]),
    ("handle.pallet_jack", "Move pallet with hand pallet jack", "two", "pallet_jack", "pallet", "S", "carry", ["OP[pump]", "OP[pull]", "BM[walk]"], {"force_n": 250}, ["push_pull"]),
    ("handle.flip_part", "Flip / turn over part", "two", "", "part", "S", "reach_place", ["GP[w1-8kg,turn]"], {}, ["wrist_deviation", "load"]),
    ("handle.slide_part", "Slide part across table", "two", "", "part", "S", "reach_place", ["GP[slide]"], {}, []),
    ("handle.stack", "Stack parts", "two", "", "part", "S", "reach_place", ["GP[w1-8kg]"], {}, ["shoulder_elevation"]),
    ("handle.kit_pick", "Kit picking from several bins", "one", "", "bin", "S", "reach_place", ["GP[d20-50] x n"], {"items": 5}, ["walking", "reach_zone"]),
    ("handle.open_carton", "Open carton with knife", "dom+sup", "box_cutter", "carton", "S", "cycle:cut_line", ["HT", "MC"], {}, ["sharp"]),
    ("handle.cut_wrap", "Cut stretch wrap from pallet", "dom+sup", "box_cutter", "pallet", "S Q", "cycle:cut_line", ["HT", "MC", "BM[bend]"], {}, ["sharp", "trunk_flexion"]),
]

C["fastening"] = [
    ("fasten.hand_start.bolt", "Hand-start bolt", "dom+sup", "", "bolt", "S K O", "cycle:finger_thread", ["GP[d<20,w<1kg,difficult]", "FL[hand,turns2-5]"], {"turns": 3}, ["pinch_grip"]),
    ("fasten.hand_start.nut", "Hand-start nut on stud", "dom+sup", "", "nut", "S K O", "cycle:finger_thread", ["GP[d<20,difficult]", "FL[hand]"], {"turns": 3}, ["pinch_grip"]),
    ("fasten.ratchet.tighten", "Tighten bolt with ratchet", "dom+sup", "ratchet_38", "bolt", "S K O Q", "cycle:ratchet", ["HT", "FL[ratchet,turns]"], {"turns": 6}, ["wrist_deviation", "shoulder_elevation"]),
    ("fasten.ratchet.backup", "Ratchet with backup wrench", "two", "ratchet_38 wrench_comb", "bolt_nut", "S K O", "cycle:ratchet", ["HT x2", "FL[ratchet]"], {"turns": 6}, ["wrist_deviation"]),
    ("fasten.wrench.open", "Tighten with open-end wrench", "dom+sup", "wrench_open", "bolt", "S K O", "cycle:wrench_swing", ["HT", "FL[wrench,turns]"], {"turns": 4}, ["wrist_deviation"]),
    ("fasten.torque.final", "Final torque with click wrench", "dom+sup", "torque_wrench", "bolt", "S K", "cycle:torque_click", ["HT", "FL[torque]"], {"torque_nm": 70}, ["force", "shoulder_elevation"]),
    ("fasten.torque.audit", "Audit torque", "dom+sup", "torque_wrench", "bolt", "S K", "cycle:torque_click", ["HT", "FL[torque]", "VC"], {"torque_nm": 70}, ["force"]),
    ("fasten.impact.drive", "Drive bolt with impact driver", "dom+sup", "impact_driver", "bolt", "S K O", "cycle:power_drive", ["HT", "OP[trigger]", "PL[tight]"], {"seconds": 2}, ["vibration", "reaction_torque"]),
    ("fasten.nutrunner", "Run nut with DC nutrunner", "dom+sup", "nutrunner", "nut", "S", "cycle:power_drive", ["HT", "OP[trigger]"], {"seconds": 2}, ["reaction_torque"]),
    ("fasten.screwdriver.manual", "Drive screw by hand", "dom+sup", "screwdriver", "screw", "S T K", "cycle:screwdriver", ["HT", "FL[screwdriver,turns]"], {"turns": 8}, ["wrist_deviation", "repetition"]),
    ("fasten.screwdriver.power", "Drive screw with power screwdriver", "dom+sup", "power_screwdriver", "screw", "S T K", "cycle:power_drive", ["HT", "OP[trigger]"], {"seconds": 1.5}, ["repetition"]),
    ("fasten.rivet.pop", "Set pop rivet", "dom+sup", "rivet_gun", "rivet", "S K O", "cycle:squeeze", ["HT", "MC[squeeze x3]"], {"squeezes": 3}, ["grip_force"]),
    ("fasten.cotter_pin", "Install cotter pin and bend legs", "dom+sup", "pliers", "pin", "S K", "cycle:squeeze", ["GP[difficult]", "HT", "MC"], {}, ["pinch_grip"]),
    ("fasten.snap_ring", "Install snap ring with pliers", "dom+sup", "snap_ring_pliers", "shaft", "S K", "cycle:squeeze", ["HT", "PL[tight]"], {}, ["grip_force"]),
    ("fasten.clip", "Push-on clip", "one", "", "clip", "S K", "operate", ["GP", "OP[apply_pressure]"], {"force_n": 40}, ["thumb"]),
    ("fasten.hold_nut", "Hold nut with wrench behind panel", "one", "wrench_comb", "nut", "S K O", "operate", ["HT", "PL[hold]"], {"seconds": 6}, ["static_hold", "reach_zone"]),
    ("fasten.torque_stripe", "Mark torque stripe", "one", "paint_pen", "bolt", "S K O", "cycle:stroke", ["HT", "MC"], {}, []),
    ("fasten.loosen", "Loosen bolt with ratchet", "dom+sup", "ratchet_38", "bolt", "S K O", "cycle:ratchet", ["HT", "FL[ratchet,loosen]"], {"turns": 4}, ["force"]),
    ("fasten.remove", "Remove fastener by hand", "one", "", "bolt", "S K O", "cycle:finger_thread", ["FL[hand]", "GP"], {"turns": 3}, []),
    ("fasten.threadlocker", "Apply threadlocker", "dom+sup", "threadlocker", "bolt", "S T", "cycle:stroke", ["HT", "MC"], {}, []),
]

C["joining"] = [
    ("join.hammer", "Strike with hammer", "dom+sup", "hammer", "part", "S K", "cycle:hammer", ["HT", "MC[strike xN]"], {"strikes": 4}, ["vibration", "shoulder_elevation", "repetition"]),
    ("join.mallet.seat", "Tap part home with mallet", "dom+sup", "mallet", "part", "S K", "cycle:hammer", ["HT", "MC[strike xN]"], {"strikes": 3, "light": True}, ["repetition"]),
    ("join.deadblow", "Seat part with dead-blow hammer", "dom+sup", "deadblow", "part", "S K", "cycle:hammer", ["HT", "MC[strike xN]"], {"strikes": 2}, ["shoulder_elevation"]),
    ("join.press_fit", "Press fit by hand", "two", "", "part", "S", "operate", ["PL[tight]", "OP[apply_pressure]"], {"force_n": 120}, ["force"]),
    ("join.punch_pin", "Drive pin with punch and hammer", "two", "punch hammer", "pin", "S K", "cycle:hammer", ["HT x2", "MC"], {"strikes": 4}, ["pinch_grip"]),
    ("join.drift_align", "Align holes with drift pin", "dom+sup", "drift_pin", "holes", "S K O", "operate", ["HT", "PL[tight]"], {}, ["force"]),
    ("join.insert_shaft", "Insert shaft / pin", "two", "", "shaft", "S", "reach_place", ["GP", "PL[tight]"], {}, []),
    ("join.slide_bushing", "Slide bushing on", "two", "", "bushing", "S", "reach_place", ["PL[tight]"], {}, []),
    ("join.snap_cover", "Snap cover in place", "two", "", "cover", "S", "operate", ["PL", "OP[apply_pressure]"], {"force_n": 60}, ["thumb"]),
    ("join.grommet", "Install grommet", "two", "", "grommet", "S K", "operate", ["GP[difficult]", "OP[apply_pressure]"], {}, ["thumb", "pinch_grip"]),
    ("join.sealant", "Apply sealant bead", "dom+sup", "caulk_gun", "seam", "S K O", "route_path", ["HT", "MC[squeeze]", "PL[path]"], {"length_cm": 60}, ["grip_force"]),
    ("join.adhesive", "Apply adhesive", "dom+sup", "glue_bottle", "surface", "S T", "route_path", ["HT", "PL[path]"], {}, []),
    ("join.label", "Peel and stick label", "two", "", "label", "S T K", "reach_place", ["GP[difficult]", "PL[tight]", "OP[smooth]"], {}, []),
    ("join.decal", "Apply large decal", "two", "squeegee", "panel", "S K", "route_path", ["PL", "HT", "MC[strokes]"], {}, ["reach_zone"]),
    ("join.clamp", "Clamp part (toggle / C-clamp)", "dom+sup", "clamp", "part", "S", "operate", ["HT", "OP[lever]"], {}, ["force"]),
]

C["wiring"] = [
    ("wire.route.frame", "Route harness along frame", "two", "", "harness", "S K Q O", "route_path", ["GP", "PL[path]", "BM"], {"length_cm": 150}, ["reach_zone", "trunk_flexion"]),
    ("wire.route.grommet", "Push harness through grommet", "two", "", "harness", "S K", "route_path", ["PL[tight]", "OP[apply_pressure]"], {}, ["pinch_grip"]),
    ("wire.connect", "Connect connector (push to click)", "two", "", "connector", "S K Q O", "operate", ["GP", "PL[tight]", "OP[apply_pressure]"], {"force_n": 40}, ["thumb", "wrist_deviation"]),
    ("wire.disconnect", "Disconnect connector", "two", "", "connector", "S K", "operate", ["OP[release]", "GP"], {}, ["pinch_grip"]),
    ("wire.tug_test", "Tug test connector latch", "two", "", "connector", "S K", "operate", ["OP[pull]", "VC"], {}, []),
    ("wire.cable_tie", "Install cable tie", "two", "", "harness", "S K O", "cycle:tie_pull", ["GP[difficult]", "MC[wrap]", "OP[pull]"], {}, ["pinch_grip", "repetition"]),
    ("wire.cut_tie", "Cut tie tail flush", "dom+sup", "flush_cutter", "tie", "S K O", "cycle:squeeze", ["HT", "MC"], {}, ["grip_force"]),
    ("wire.strip", "Strip wire", "dom+sup", "wire_stripper", "wire", "S T", "cycle:squeeze", ["HT", "MC[squeeze,pull]"], {}, ["grip_force", "repetition"]),
    ("wire.crimp", "Crimp terminal", "dom+sup", "crimper", "terminal", "S T", "cycle:squeeze", ["HT", "PL[tight]", "MC[squeeze]"], {"force_n": 200}, ["grip_force"]),
    ("wire.insert_terminal", "Insert terminal into housing", "two", "", "housing", "S T", "reach_place", ["GP[difficult]", "PL[tight]"], {}, ["pinch_grip"]),
    ("wire.heat_shrink", "Shrink sleeve with heat gun", "dom+sup", "heat_gun", "wire", "S T", "cycle:sweep", ["HT", "MC[sweep]"], {"seconds": 6}, ["burn"]),
    ("wire.battery", "Connect battery cable", "dom+sup", "wrench_comb", "battery", "S Q", "cycle:wrench_swing", ["PL[tight]", "HT", "FL"], {}, ["trunk_flexion"]),
    ("wire.ground_strap", "Attach ground strap", "dom+sup", "ratchet_38", "stud", "S K", "cycle:ratchet", ["PL", "HT", "FL"], {}, []),
    ("wire.terminal_block", "Terminate wire on terminal block", "dom+sup", "screwdriver_small", "terminal", "S T", "cycle:screwdriver", ["GP[difficult]", "PL[tight]", "HT", "FL"], {"turns": 4}, ["wrist_deviation"]),
    ("wire.p_clip", "Install P-clip with screw", "two", "power_screwdriver", "harness", "S K O", "cycle:power_drive", ["PL[hold]", "HT", "OP[trigger]"], {}, ["static_hold"]),
    ("wire.label", "Label wire", "two", "", "wire", "S T", "reach_place", ["GP[difficult]", "PL"], {}, []),
    ("wire.loom", "Install split loom", "two", "", "harness", "S K", "route_path", ["GP", "MC[push along]"], {"length_cm": 80}, ["repetition"]),
    ("wire.tray", "Lay cable in tray", "two", "", "tray", "S P", "route_path", ["PL[path]"], {"length_cm": 200}, ["shoulder_elevation"]),
    ("wire.coil", "Coil cable", "two", "", "cable", "S", "cycle:coil", ["MC[loops]"], {"loops": 6}, ["repetition"]),
    ("wire.multimeter", "Check continuity with multimeter", "two", "multimeter", "connector", "S T", "inspect", ["HT", "PL[probe]", "VC"], {}, []),
]

C["hydraulics"] = [
    ("hyd.hose.connect", "Connect hose fitting", "dom+sup", "wrench_open", "fitting", "S K Q", "cycle:wrench_swing", ["GP", "FL[hand]", "HT", "FL[wrench]"], {"turns": 3}, ["wrist_deviation"]),
    ("hyd.jic.two_wrench", "Tighten JIC fitting with two wrenches", "two", "wrench_open wrench_open", "fitting", "S K", "cycle:wrench_swing", ["HT x2", "FL"], {}, ["force"]),
    ("hyd.oring", "Install O-ring", "two", "", "fitting", "S T", "reach_place", ["GP[difficult]", "PL[tight]"], {}, ["pinch_grip"]),
    ("hyd.fill", "Fill hydraulic oil", "two", "oil_jug", "tank", "S", "operate", ["GP[w8-22kg]", "PL[hold]"], {"seconds": 20}, ["static_hold", "load"]),
    ("hyd.level_check", "Check oil level", "one", "", "tank", "S Q", "inspect", ["VC", "BM[bend]"], {}, ["trunk_flexion"]),
    ("hyd.bleed", "Bleed hydraulic system", "dom+sup", "wrench_open", "valve", "S K", "cycle:wrench_swing", ["HT", "FL[loosen]", "VC"], {}, []),
    ("hyd.cap", "Cap / plug port", "one", "", "port", "S K", "cycle:finger_thread", ["GP", "FL[hand]"], {}, []),
    ("hyd.wipe", "Wipe spill", "one", "rag", "surface", "S K Q", "cycle:wipe", ["HT", "MC[strokes]"], {}, ["trunk_flexion"]),
    ("hyd.filter", "Install filter", "two", "", "filter", "S Q", "cycle:finger_thread", ["GP", "FL[hand,turns]"], {"turns": 4}, ["trunk_flexion"]),
    ("hyd.quick_coupler", "Connect quick coupler", "two", "", "coupler", "S Q", "operate", ["PL[tight]", "OP[apply_pressure]"], {}, ["thumb"]),
]

C["positioning"] = [
    ("pos.align.two_hands", "Align part with both hands", "two", "", "part", "S K O", "reach_place", ["PL[tight]"], {}, ["static_hold"]),
    ("pos.hold_for_partner", "Hold part while partner fastens", "two", "", "part", "S K O", "team", ["PL[hold]"], {"seconds": 15}, ["static_hold", "coordination"]),
    ("pos.hang_on_hooks", "Lift and hang part on hooks", "two", "", "part", "S", "reach_place", ["GP[w8-22kg]", "PL"], {}, ["shoulder_elevation", "niosh"]),
    ("pos.guide_crane_load", "Guide crane load by hand", "two", "", "crane_load", "S", "team", ["OP[push]", "VC"], {}, ["suspended_load"]),
    ("pos.insert_slot", "Insert part into slot", "two", "", "slot", "S", "reach_place", ["PL[tight]"], {}, []),
    ("pos.cover_panel", "Install cover panel", "two", "", "panel", "S K", "reach_place", ["GP[w1-8kg]", "PL[tight]"], {}, ["reach_zone"]),
    ("pos.door_hinge", "Hang door / gate on hinges", "two", "", "gate", "S", "reach_place", ["GP[w8-22kg]", "PL[tight]"], {}, ["load", "static_hold"]),
    ("pos.rail_team", "Install guardrail (two people)", "team", "", "rail", "S P", "team", ["GP[team]", "PL[tight]"], {}, ["coordination", "shoulder_elevation"]),
    ("pos.lower_on_pins", "Lower part onto locating pins", "two", "", "part", "S", "reach_place", ["PL[tight]"], {}, []),
    ("pos.jig", "Load part into alignment jig", "two", "", "jig", "S", "reach_place", ["GP", "PL[tight]", "OP[lever]"], {}, []),
    ("pos.rotate_fixture", "Rotate subassembly on fixture", "two", "", "fixture", "S", "operate", ["OP[turn]"], {"force_n": 100}, ["force", "twist_with_load"]),
    ("pos.stands", "Set part on stands", "two", "", "stands", "S", "reach_place", ["GP[w8-22kg]", "PL"], {}, ["niosh"]),
]

C["equipment"] = [
    ("equip.crane.pendant", "Operate crane pendant (lift / travel)", "one", "pendant", "crane", "S", "operate", ["OP[button]", "VC", "BM[walk]"], {"seconds": 10}, ["suspended_load"]),
    ("equip.crane.hook_on", "Attach sling / hook to load", "two", "sling", "crane_load", "S Q", "reach_place", ["GP", "PL[tight]"], {}, ["trunk_flexion"]),
    ("equip.crane.hook_off", "Unhook load", "two", "", "crane_load", "S Q", "reach_place", ["GP", "PL"], {}, []),
    ("equip.lift_table.raise", "Raise / lower lift table", "one", "", "lift_table", "S", "operate", ["OP[button hold]"], {"seconds": 8}, []),
    ("equip.unit.drive_test", "Drive / function-test machine from platform", "two", "controller", "machine", "S", "operate", ["OP[joystick]", "VC"], {"seconds": 30}, []),
    ("equip.key_switch", "Turn key switch", "one", "key", "switch", "S", "operate", ["OP[turn]"], {}, []),
    ("equip.estop", "Press / reset E-stop", "one", "", "estop", "S", "operate", ["OP[button]"], {}, []),
    ("equip.foot_pedal", "Operate foot pedal", "none", "", "pedal", "S T", "operate", ["OP[foot]"], {}, ["standing_one_leg"]),
    ("equip.scanner", "Scan barcode", "one", "scanner", "label", "S", "operate", ["HT", "OP[trigger]", "VC"], {}, []),
    ("equip.tablet", "Enter data on tablet / terminal", "two", "tablet", "terminal", "S T", "operate", ["OP[keys]", "VC"], {"seconds": 8}, ["neck_flexion"]),
    ("equip.print_label", "Print and take label", "one", "", "printer", "S", "reach_place", ["OP[button]", "GP"], {}, []),
    ("equip.conveyor", "Start / stop conveyor", "one", "", "conveyor", "S", "operate", ["OP[button]"], {}, []),
    ("equip.guard_door", "Open / close guard door", "one", "", "door", "S", "operate", ["OP[pull]"], {}, []),
    ("equip.scale_step", "Weigh machine on platform scale (read display)", "none", "", "scale", "S", "inspect", ["VC", "BM[walk]"], {}, []),
    ("equip.step_stand", "Climb onto / off step stand", "two", "step_stand", "stand", "S P", "body", ["BM[climb]"], {"steps": 2}, ["fall"]),
    ("equip.move_step_stand", "Move step stand", "two", "step_stand", "stand", "S", "carry", ["GP[w8-22kg]", "BM[walk]"], {}, ["load"]),
    ("equip.air_hose", "Connect / disconnect air hose", "two", "", "coupler", "S", "operate", ["PL", "OP[sleeve]"], {}, []),
]

C["inspection"] = [
    ("inspect.walk_around", "Visual walk-around inspection", "none", "", "machine", "S Q", "inspect", ["VC", "BM[walk]", "BM[bend]"], {}, ["walking"]),
    ("inspect.underside", "Inspect underside with mirror / light", "two", "flashlight mirror", "machine", "S Q K", "inspect", ["HT x2", "VC", "BM[squat]"], {}, ["neck_extension", "knee_flexion"]),
    ("inspect.tape", "Measure with tape", "two", "tape_measure", "part", "S K", "inspect", ["HT", "PL", "VC"], {}, []),
    ("inspect.caliper", "Measure with caliper", "two", "caliper", "part", "S T", "inspect", ["HT", "PL[tight]", "VC"], {}, []),
    ("inspect.gauge", "Check with go / no-go gauge", "dom+sup", "gauge", "hole", "S T", "inspect", ["HT", "PL[tight]"], {}, []),
    ("inspect.pull_test", "Pull test", "two", "", "part", "S", "operate", ["OP[pull]"], {"force_n": 100}, ["force"]),
    ("inspect.checklist", "Sign checklist", "one", "pen", "document", "S T", "operate", ["HT", "MC[write]", "VC"], {}, ["neck_flexion"]),
    ("inspect.photo", "Photograph defect with tablet", "two", "tablet", "part", "S", "inspect", ["HT", "OP[button]", "VC"], {}, []),
    ("inspect.tag", "Attach hold / inspection tag", "two", "tag", "part", "S", "reach_place", ["GP[difficult]", "PL"], {}, []),
    ("inspect.leak_check", "Check fittings for leaks", "one", "rag", "fitting", "S Q K", "inspect", ["VC", "HT", "BM[squat]"], {}, ["knee_flexion"]),
]

C["finishing"] = [
    ("finish.grind", "Grind weld with angle grinder", "two", "angle_grinder", "weld", "S K", "cycle:sweep", ["HT", "OP[trigger]", "MC[strokes]"], {"seconds": 15}, ["vibration", "line_of_fire"]),
    ("finish.sand", "Sand edge", "dom+sup", "sanding_block", "edge", "S K", "cycle:wipe", ["HT", "MC[strokes]"], {"strokes": 12}, ["repetition"]),
    ("finish.file", "Deburr with file", "two", "file", "edge", "S T", "cycle:wipe", ["HT", "MC[strokes]"], {"strokes": 8}, ["repetition"]),
    ("finish.touch_up", "Paint touch-up", "dom+sup", "paint_brush", "surface", "S K", "cycle:stroke", ["HT", "MC[strokes]"], {}, []),
    ("finish.spray", "Spray paint / aerosol", "dom+sup", "spray_can", "surface", "S K", "cycle:sweep", ["HT", "OP[trigger]", "MC[sweep]"], {"seconds": 6}, ["fumes"]),
    ("finish.wipe", "Wipe clean with cloth", "one", "rag", "surface", "S K", "cycle:wipe", ["HT", "MC[strokes]"], {"strokes": 6}, []),
    ("finish.blow_off", "Blow off with air gun", "one", "air_gun", "surface", "S", "cycle:sweep", ["HT", "OP[trigger]"], {"seconds": 4}, ["line_of_fire"]),
    ("finish.grease", "Grease zerk with grease gun", "two", "grease_gun", "zerk", "S K Q", "cycle:squeeze", ["HT", "PL[tight]", "MC[pump x3]"], {"pumps": 3}, ["grip_force", "trunk_flexion"]),
    ("finish.lube_spray", "Spray lubricant", "one", "spray_can", "part", "S K", "operate", ["HT", "OP[trigger]"], {}, []),
    ("finish.sweep", "Sweep work area", "two", "broom", "floor", "S", "cycle:sweep", ["HT", "MC[strokes]", "BM[walk]"], {"seconds": 20}, ["trunk_flexion"]),
]

C["posture"] = [
    ("work.waist", "Work at waist height (reference)", "two", "", "point", "S", "operate", [], {}, []),
    ("work.overhead", "Work overhead", "two", "", "point", "S P", "operate", [], {}, ["shoulder_elevation", "neck_extension"]),
    ("work.kneeling", "Work kneeling", "two", "", "point", "K", "operate", [], {}, ["knee_contact"]),
    ("work.lying", "Work lying on creeper under machine", "two", "creeper", "point", "L", "operate", [], {}, ["neck_extension", "shoulder_elevation"]),
    ("work.reach_across", "Work reaching across", "two", "", "point", "S", "operate", [], {}, ["reach_zone", "trunk_flexion"]),
    ("work.squatting", "Work squatting", "two", "", "point", "Q", "operate", [], {}, ["knee_flexion"]),
    ("work.step_stand", "Work on step stand", "two", "step_stand", "point", "P", "operate", [], {}, ["fall"]),
    ("work.lean_in", "Lean into machine", "two", "", "point", "S", "operate", [], {}, ["trunk_flexion", "contact_stress"]),
    ("work.seated", "Work seated on stool", "two", "stool", "point", "T", "operate", [], {}, []),
    ("work.one_foot_raised", "Work with one foot on rail", "two", "", "point", "S", "operate", [], {}, []),
]

C["walking"] = [
    ("walk.path", "Walk to location", "none", "", "point", "S", "walk", ["BM[walk per m]"], {}, ["walking", "traffic"]),
    ("walk.tote", "Walk carrying tote", "two", "", "tote", "S", "carry", ["BM[walk]"], {}, ["walking", "load"]),
    ("walk.turn180", "Turn around", "none", "", "direction", "S", "walk", ["BM[turn]"], {"angle_deg": 180}, []),
    ("walk.start_stop", "Start / stop walking", "none", "", "point", "S", "walk", ["BM[walk]"], {}, []),
    ("walk.sidestep_line", "Sidestep along the line", "none", "", "point", "S", "walk", ["BM[step]"], {}, []),
    ("walk.wait", "Wait / let coworker pass", "none", "", "point", "S", "walk", [], {"seconds": 2}, ["waiting"]),
    ("walk.ramp", "Walk up / down ramp", "none", "", "ramp", "S", "walk", ["BM[walk]"], {}, ["slip"]),
    ("walk.stairs", "Climb platform stairs", "one", "", "stairs", "S", "walk", ["BM[climb]"], {}, ["fall"]),
]

CATEGORY_TITLES = {"basic": "Basic elements (MTM-UAS building blocks)", "handling": "Material handling",
                   "fastening": "Fastening", "joining": "Joining and fitting", "wiring": "Wiring and electrical",
                   "hydraulics": "Hydraulics and fluids", "positioning": "Positioning and assembly",
                   "equipment": "Operating equipment", "inspection": "Inspection and test",
                   "finishing": "Finishing and cleaning", "posture": "Work postures", "walking": "Walking"}
POSTURE = {"S": "standing", "K": "kneeling", "Q": "squatting", "O": "overhead", "T": "seated", "L": "lying",
           "P": "on_stand"}


def q(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, list):
        return "[" + ", ".join(q(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{k} = {q(x)}" for k, x in v.items()) + " }" if v else "{}"
    return '"' + str(v).replace('"', '\\"') + '"'


def main():
    lines = ['# Manufacturing motion catalog for frt_core.people (one [[motion]] per movement).',
             '# Generated once by tools/make_motion_catalog.py; edit freely - this file is the source of truth.',
             '# Fields: see docs/people/DESIGN.md. uas = MTM-UAS building blocks (times come from the plant card).',
             'version = 1', '']
    n = 0
    for cat, rows in C.items():
        lines.append(f"# ---------------------------------------------------------------- {CATEGORY_TITLES[cat]}")
        for (mid, name, hands, tool, obj, post, pattern, uas, params, risks) in rows:
            lines += ["[[motion]]", f"id = {q(mid)}", f"name = {q(name)}", f"category = {q(cat)}",
                      f"hands = {q(hands)}", f"tools = {q(tool.split() if tool else [])}", f"object = {q(obj)}",
                      f"postures = {q([POSTURE[c] for c in post.split()])}", f"pattern = {q(pattern)}",
                      f"uas = {q(uas)}", f"params = {q(params)}", f"risks = {q(risks)}", ""]
            n += 1
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"{n} motions -> {OUT}")


if __name__ == "__main__":
    main()
