"""Material library: industrial looks as OmniPBR (MDL, for RTX) with a UsdPreviewSurface fallback (Storm, other
renderers, previews). Textures are projected in object space, so meshes without UVs (generators, CAD) still get grain.

    lib = MaterialLibrary(stage, "/World/Looks")
    ref = lib.get("powder_yellow")        # MatRef(material, rgb)

Colours are sRGB as people quote them (RAL references in the comments) and converted to linear for rendering.
"""
import os
from collections import namedtuple

from pxr import Gf, Sdf, UsdShade

MatRef = namedtuple("MatRef", "material rgb")

NV_MATS = os.environ.get("FRT_NVIDIA_MATERIALS",
                         r"C:/Nvidia_Assets/Warehouse_NVD/Assets/DigitalTwin/Materials/Base/")
TEX = {
    "paint_n": "Paint/Painted_Metal/T_Painted_Metal_Green_A_Normal.png",
    "powder_n": "Paint/Painted_Metal/T_Metal_Gray_PowderCoated_A_Normal.png",
    "grunge_r": "Paint/Misc/T_Grunge_Roughness.png",
    "satin_n": "Paint/Paint_Satin/Paint_Satin_N.png",
    "rubber_n": "Rubber/Rough/T_Rubber_Black_Rough_A_Normal.png",
    "steel_n": "Metals/Steel/T_Steel_A1_Normal.png",
    "grainy_n": "Plastics/Grainy/T_Plastic_Grainy_A1_Normal.png",     # moulded ABS / PA grain
    "plastic_n": "Plastics/Misc/T_Plastic_Normal.png",
    "rubber_new_n": "Rubber/New/T_Rubber_New_A1_Normal.png",
}
# self-lit looks: pilot lamps, LEDs, light fittings (OmniPBR emission, intensity in nits-ish units)
EMISSIVE = {"light_emit": 4000.0, "lamp_orange": 900.0, "lamp_green": 700.0, "lamp_white": 900.0}


def srgb_to_linear(c):
    return tuple(((v + 0.055) / 1.055) ** 2.4 if v > 0.04045 else v / 12.92 for v in c)


def _rgb(hexstr):
    h = hexstr.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


# name: (sRGB colour, roughness, metallic, normal map, bump, roughness-map influence)
LOOKS = {
    "powder_yellow": (_rgb("#e3a700"), 0.50, 0.0, "powder_n", 0.5, 0.2),     # signal yellow (RAL 1003/1023, as it looks)
    "powder_orange": (_rgb("#e25303"), 0.42, 0.0, "powder_n", 0.5, 0.2),     # RAL 2004 pure orange
    "powder_blue": (_rgb("#0e518d"), 0.40, 0.0, "powder_n", 0.5, 0.2),       # RAL 5010 gentian blue
    "powder_green": (_rgb("#2d572c"), 0.42, 0.0, "powder_n", 0.5, 0.2),      # RAL 6001 emerald green
    "powder_red": (_rgb("#a52019"), 0.42, 0.0, "powder_n", 0.5, 0.2),        # RAL 3020 traffic red
    "powder_lightgrey": (_rgb("#d7d7d7"), 0.45, 0.0, "powder_n", 0.4, 0.2),  # RAL 7035 light grey
    "powder_grey": (_rgb("#8f9695"), 0.45, 0.0, "powder_n", 0.4, 0.2),       # RAL 7042 traffic grey A
    "powder_dark": (_rgb("#383e42"), 0.45, 0.0, "powder_n", 0.5, 0.2),       # RAL 7016 anthracite
    "powder_black": (_rgb("#151618"), 0.50, 0.0, "powder_n", 0.5, 0.2),      # RAL 9005 jet black (hazard bands)
    "powder_anthracite": (_rgb("#3a3f44"), 0.48, 0.0, "powder_n", 0.5, 0.2),  # RAL 7016: motors, gear cases
    "galvanised": ((0.62, 0.63, 0.64), 0.38, 1.0, "steel_n", 0.15, 0.0),
    "stainless": ((0.72, 0.72, 0.73), 0.25, 1.0, "steel_n", 0.1, 0.0),
    "aluminium": ((0.80, 0.81, 0.82), 0.32, 1.0, "steel_n", 0.1, 0.0),
    "steel_dark": ((0.30, 0.31, 0.32), 0.40, 1.0, "steel_n", 0.2, 0.0),
    "chrome": ((0.85, 0.86, 0.87), 0.12, 1.0, None, 0.0, 0.0),
    "rubber_black": ((0.05, 0.05, 0.05), 0.85, 0.0, "rubber_n", 0.8, 0.0),
    "belt_black": ((0.08, 0.08, 0.085), 0.75, 0.0, "rubber_n", 0.6, 0.0),
    "plastic_black": ((0.06, 0.06, 0.065), 0.45, 0.0, "grainy_n", 0.35, 0.2),
    "plastic_blue": (_rgb("#1f5fae"), 0.40, 0.0, None, 0.0, 0.0),
    "plastic_grey": (_rgb("#6d7377"), 0.45, 0.0, None, 0.0, 0.0),
    "wood_top": (_rgb("#c9a170"), 0.60, 0.0, "satin_n", 0.4, 0.2),          # birch / beech worktop
    "esd_mat": (_rgb("#5f7a6e"), 0.70, 0.0, "satin_n", 0.3, 0.0),           # grey-green ESD mat
    "floor_yellow": (_rgb("#f2c40c"), 0.55, 0.0, "satin_n", 0.2, 0.0),      # floor tape / paint
    "floor_white": (_rgb("#ececec"), 0.55, 0.0, "satin_n", 0.2, 0.0),
    "floor_green": (_rgb("#3c8a4a"), 0.55, 0.0, "satin_n", 0.2, 0.0),
    "floor_red": (_rgb("#c0281e"), 0.55, 0.0, "satin_n", 0.2, 0.0),
    "cardboard": (_rgb("#b8915f"), 0.80, 0.0, "satin_n", 0.4, 0.3),
    "tape_brown": (_rgb("#9c7a4a"), 0.50, 0.0, None, 0.0, 0.0),
    "steel_tread": ((0.45, 0.46, 0.48), 0.45, 1.0, "steel_n", 0.5, 0.0),
    "stretch_wrap": (_rgb("#d6dee2"), 0.18, 0.0, None, 0.0, 0.0),
    "light_emit": ((1.0, 1.0, 0.97), 0.30, 0.0, None, 0.0, 0.0),
    # finishes for detailed parts
    "bolt_zinc": ((0.70, 0.71, 0.72), 0.30, 1.0, "steel_n", 0.1, 0.0),       # zinc-plated fasteners
    "steel_machined": ((0.62, 0.63, 0.65), 0.22, 1.0, "steel_n", 0.1, 0.0),  # wheels, axles, shafts
    "steel_blued": ((0.18, 0.19, 0.21), 0.35, 1.0, "steel_n", 0.2, 0.0),     # load chain
    "cast_aluminium": ((0.66, 0.67, 0.68), 0.45, 1.0, "steel_n", 0.3, 0.0),  # motor / gear housings
    "cable_pvc": ((0.04, 0.04, 0.045), 0.35, 0.0, "rubber_n", 0.3, 0.0),     # cable sheath
    "concrete": (_rgb("#a6a39b"), 0.85, 0.0, None, 0.0, 0.0),                # grout, plinths
    "abs_lightgrey": (_rgb("#d4d5d0"), 0.42, 0.0, "grainy_n", 0.3, 0.25),   # RAL 7035 ABS / PC enclosure
    "powder_black_tex": (_rgb("#1a1b1d"), 0.62, 0.0, "powder_n", 1.4, 0.35),  # textured powder coat (handles)
    "grip_rubber": (_rgb("#161616"), 0.80, 0.0, "rubber_new_n", 1.0, 0.3),    # TPE handle grips
    "hose_rubber": (_rgb("#101010"), 0.62, 0.0, "rubber_n", 0.6, 0.25),       # hydraulic hose cover
    "brass": (_rgb("#b59a52"), 0.32, 1.0, "steel_n", 0.15, 0.1),              # fittings, valve bodies
    "label_vinyl": (_rgb("#f1f1ee"), 0.28, 0.0, None, 0.0, 0.0),              # printed vinyl legends
    "lamp_orange": (_rgb("#ff8a1e"), 0.20, 0.0, None, 0.0, 0.0),
    "lamp_green": (_rgb("#35d45a"), 0.20, 0.0, None, 0.0, 0.0),
    "lamp_white": (_rgb("#fff6e8"), 0.20, 0.0, None, 0.0, 0.0),
    "hole": (_rgb("#0a0b0c"), 0.95, 0.0, None, 0.0, 0.0),                    # punched holes, recesses (seen as shadow)
    "nylon_white": (_rgb("#dcd9cf"), 0.38, 0.0, None, 0.0, 0.0),             # flow wheels, ball cages
    "plastic_red": (_rgb("#b3261e"), 0.30, 0.0, "plastic_n", 0.15, 0.0),     # glossy moulded (E-stop)
    "plastic_yellow": (_rgb("#e0a800"), 0.40, 0.0, None, 0.0, 0.0),
}

# Textured NVIDIA presets (OmniPBR with albedo / ORM / normal sets) used for RTX when the asset packs are installed;
# meshes carry box-projected UVs in metres (geom.Shapes), so the textures land at real scale on any part.
PRESETS = {
    "powder_yellow": "Paint/Painted_Metal_Yellow_A.mdl",
    "powder_orange": "Paint/Painted_Metal_Orange_A.mdl",
    "powder_blue": "Paint/Painted_Metal_Blue_A.mdl",
    "powder_green": "Paint/Painted_Metal_Green_A.mdl",
    "powder_red": "Paint/Painted_Metal_Red_A.mdl",
    "powder_grey": "Paint/Painted_Metal_Grey_Glossy_A.mdl",
    "powder_lightgrey": "Paint/Painted_Metal_White_Rough_A.mdl",
    "powder_dark": "Paint/Powdercoat_Black_A.mdl",
    "powder_black": "Paint/Painted_Metal_Black_Glossy_A.mdl",
    "powder_anthracite": "Paint/Painted_Metal_Grey_Glossy_A.mdl",
    "galvanised": "Metals/Steel_MidGray_Matte_A.mdl",
    "steel_tread": "Metals/Metal_Steel_Textured_A.mdl",          # diamond tread plate
    "stainless": "Metals/Steel_Stainless_Brushed_A.mdl",
    "aluminium": "Metals/Aluminium_Brushed_A.mdl",
    "steel_dark": "Metals/Iron_A.mdl",
    "chrome": "Metals/Chrome_A.mdl",
    "rubber_black": "Rubber/Rubber_Black_Dirty_A.mdl",
    "belt_black": "Rubber/Rubber_Black_Lined_A.mdl",
    "bolt_zinc": "Metals/Steel_A.mdl",
    "steel_machined": "Metals/Steel_Bright_A.mdl",
    "steel_blued": "Metals/Steel_Blued_A.mdl",
    "cast_aluminium": "Metals/Aluminum_Cast_A.mdl",
    "cable_pvc": "Rubber/Rubber_Black_Glossy_A.mdl",
    "concrete": "Concrete/Concrete_Smooth_A.mdl",
    "brass": "Metals/Brass_A.mdl",
    "wood_top": "Wood/Plywood_A_Tiling.mdl",
}


# preset tweaks: painted steel is semi-matte powder coat in its real colour. The presets brighten their albedo a
# lot (albedo_add 0.35-0.65) and tint a near-white base, which reads as plastic; use our colour, nothing added.
PAINT_TUNE = {"reflection_roughness_constant": 0.55, "reflection_roughness_texture_influence": 0.35,
              "albedo_add": 0.0, "albedo_brightness": 1.0, "albedo_desaturation": 0.0, "specular_level": 0.45}
METAL_TUNE = {"albedo_brightness": 0.85}
PRESET_TUNE = {name: (PAINT_TUNE if path.startswith("Paint/") else METAL_TUNE if path.startswith("Metals/") else {})
               for name, path in PRESETS.items()}


class MaterialLibrary:
    def __init__(self, stage, root="/World/Looks", textures=NV_MATS, presets=True):
        self.stage, self.root, self.textures = stage, root.rstrip("/"), textures
        self.presets = presets
        self._cache = {}

    def names(self):
        return sorted(LOOKS)

    def get(self, name):
        if name in self._cache:
            return self._cache[name]
        if name not in LOOKS:
            raise KeyError(f"unknown look {name!r}; known: {', '.join(sorted(LOOKS))}")
        srgb, rough, metal, normal, bump, rmap = LOOKS[name]
        lin = srgb_to_linear(srgb)
        path = f"{self.root}/{name}"
        mat = UsdShade.Material.Define(self.stage, path)
        sh = UsdShade.Shader.Define(self.stage, path + "/OmniPBR")
        C3, F, B, A, F2 = (Sdf.ValueTypeNames.Color3f, Sdf.ValueTypeNames.Float, Sdf.ValueTypeNames.Bool,
                           Sdf.ValueTypeNames.Asset, Sdf.ValueTypeNames.Float2)
        preset = PRESETS.get(name) if self.presets else None
        if preset and self.textures and os.path.isfile(self.textures + preset):
            # the textured preset as it ships (its texture paths are relative to the .mdl); UVs come from the mesh
            sh.SetSourceAsset(Sdf.AssetPath(self.textures + preset), "mdl")
            sh.SetSourceAssetSubIdentifier(os.path.splitext(os.path.basename(preset))[0], "mdl")
            sh.CreateInput("project_uvw", B).Set(False)
            for k, v in PRESET_TUNE.get(name, {}).items():
                sh.CreateInput(k, F).Set(float(v))
            if PRESETS[name].startswith("Paint/"):
                sh.CreateInput("diffuse_tint", C3).Set(Gf.Vec3f(*lin))
            out = sh.CreateOutput("out", Sdf.ValueTypeNames.Token)
            for o in (mat.CreateSurfaceOutput("mdl"), mat.CreateDisplacementOutput("mdl"), mat.CreateVolumeOutput("mdl")):
                o.ConnectToSource(out)
            self._preview(mat, path, lin, rough, metal, C3, F)
            ref = MatRef(mat, srgb)
            self._cache[name] = ref
            return ref
        sh.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
        sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
        sh.CreateInput("diffuse_color_constant", C3).Set(Gf.Vec3f(*lin))
        sh.CreateInput("reflection_roughness_constant", F).Set(float(rough))
        sh.CreateInput("metallic_constant", F).Set(float(metal))
        if name in EMISSIVE:
            sh.CreateInput("enable_emission", B).Set(True)
            sh.CreateInput("emissive_color", C3).Set(Gf.Vec3f(*lin))
            sh.CreateInput("emissive_intensity", F).Set(EMISSIVE[name])
        tex_ok = self.textures and os.path.isdir(self.textures)
        if normal and tex_ok:
            inp = sh.CreateInput("normalmap_texture", A)
            inp.Set(Sdf.AssetPath(self.textures + TEX[normal]))
            inp.GetAttr().SetColorSpace("raw")
            sh.CreateInput("bump_factor", F).Set(float(bump))
        if rmap > 0 and tex_ok:
            inp = sh.CreateInput("reflectionroughness_texture", A)
            inp.Set(Sdf.AssetPath(self.textures + TEX["grunge_r"]))
            inp.GetAttr().SetColorSpace("raw")
            sh.CreateInput("reflection_roughness_texture_influence", F).Set(float(rmap))
        sh.CreateInput("project_uvw", B).Set(True)
        sh.CreateInput("world_or_object", B).Set(False)
        sh.CreateInput("texture_scale", F2).Set(Gf.Vec2f(0.05, 0.05))
        out = sh.CreateOutput("out", Sdf.ValueTypeNames.Token)
        mat.CreateSurfaceOutput("mdl").ConnectToSource(out)
        mat.CreateDisplacementOutput("mdl").ConnectToSource(out)
        mat.CreateVolumeOutput("mdl").ConnectToSource(out)
        pv = self._preview(mat, path, lin, rough, metal, C3, F)
        if name in EMISSIVE:
            pv.CreateInput("emissiveColor", C3).Set(Gf.Vec3f(*lin))
        ref = MatRef(mat, srgb)
        self._cache[name] = ref
        return ref

    def _preview(self, mat, path, lin, rough, metal, C3, F):
        """UsdPreviewSurface fallback (Storm, other renderers, quick previews)."""
        pv = UsdShade.Shader.Define(self.stage, path + "/Preview")
        pv.CreateIdAttr("UsdPreviewSurface")
        pv.CreateInput("diffuseColor", C3).Set(Gf.Vec3f(*lin))
        pv.CreateInput("roughness", F).Set(float(rough))
        pv.CreateInput("metallic", F).Set(float(metal))
        mat.CreateSurfaceOutput().ConnectToSource(pv.ConnectableAPI(), "surface")
        return pv

    def refs(self, names):
        """{name: MatRef} for the generator mesh builder."""
        return {n: self.get(n) for n in names}
