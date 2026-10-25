"""Kimodo side of the job pipeline (runs in the Kimodo env): generate a motion that meets our end-effector keyframes.

Input npz (from tools/kimodo/kimodo_job.py, toolkit side):
    seconds, fps_in; frames [K] (our frame index), t [K] (s);
    names [Jc] canonical joint names; P [K, Jc, 3] world positions (cm, our worker frame); R [K, Jc, 3, 3] world rots
    use_r, use_l, use_feet [K] bool: which effectors to constrain at each keyframe
    scale: cm -> Kimodo metres for this body (Kimodo hip height / ours / 100)
Output npz: posed_joints [N, 77, 3] (metres, Kimodo frame), names77, fps.

Our canonical frame and Kimodo's agree (Y up, facing +Z, left = +X, T-pose rest with palms down), so rotations
carry over unchanged and positions only scale.
"""
import os
import sys

sys.path.insert(0, os.environ.get("KIMODO_SRC", r"C:/OmniverseData/kimodo/src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np   # noqa: E402
import torch   # noqa: E402

# canonical (ours) -> SOMA joint names
TO_SOMA = {"pelvis": "Hips", "spine_01": "Spine1", "spine_02": "Spine2", "spine_03": "Chest", "neck_01": "Neck1",
           "head": "Head"}
for _s, _S in (("l", "Left"), ("r", "Right")):
    TO_SOMA.update({f"clavicle_{_s}": f"{_S}Shoulder", f"upperarm_{_s}": f"{_S}Arm", f"lowerarm_{_s}": f"{_S}ForeArm",
                    f"hand_{_s}": f"{_S}Hand", f"thigh_{_s}": f"{_S}Leg", f"calf_{_s}": f"{_S}Shin",
                    f"foot_{_s}": f"{_S}Foot", f"ball_{_s}": f"{_S}ToeBase"})


TEXT = os.environ.get("KIMODO_TEXT", r"C:/OmniverseData/kimodo/runs/text/kimodo_text.npz")


class Cached:
    """Kimodo's text encoder from features computed once elsewhere (tools/kimodo/cloud/encode_prompts.py: the 8B
    encoder does not fit this machine): a prompt in the cache gets its features, any other text (and "") none -
    Kimodo's own handling of an empty prompt."""

    def __init__(self, dim, device):
        import json
        self.dim, self.device, self.feat = dim, device, {}
        if os.path.isfile(TEXT):
            d = np.load(TEXT)
            texts = json.loads(str(d["__texts__"]))
            self.feat = {t.strip(): d[k].reshape(1, -1) for k, t in texts.items() if k in d}
        self.used = 0

    def __call__(self, texts):
        out = []
        for t in texts:
            f = self.feat.get((t or "").strip())
            self.used += f is not None
            out.append(torch.tensor(f, dtype=torch.float32) if f is not None else torch.zeros((1, self.dim)))
        return torch.stack(out).to(self.device), [1 if (t or "").strip() in self.feat else 0 for t in texts]

    def to(self, *a, **k):
        return self


def main():
    from kimodo import load_model
    from kimodo.constraints import EndEffectorConstraintSet
    from kimodo.skeleton.registry import build_skeleton
    from kimodo_run import NoText, text_dim
    from kimodo.model.load_model import _resolve_hf_model_path
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    name = "kimodo-soma-rp-v1.2"
    path = _resolve_hf_model_path(name)
    model = load_model(name, device=device, text_encoder=Cached(text_dim(str(path)), device))
    pairs = list(zip(sys.argv[1::2], sys.argv[2::2]))       # several jobs per call: the model loads once
    for src, out in pairs:
        try:
            generate(model, device, src, out)
        except Exception as e:                                # one bad job does not stop the batch
            print("failed", src, type(e).__name__, e)


def generate(model, device, src, out):
    from kimodo.constraints import EndEffectorConstraintSet
    from kimodo.skeleton.registry import build_skeleton
    d = np.load(src, allow_pickle=True)
    sk = model.skeleton                                   # the model's working skeleton (somaskel30)
    names = [n for n, _ in sk.bone_order_names_with_parents]
    neutral = sk.neutral_joints.cpu().numpy()             # rest (T-pose) joint positions, hips at the origin
    fps = float(model.fps)
    n = int(round(float(d["seconds"]) * fps))
    cn = [str(x) for x in d["names"]]
    P, R = d["P"].astype(np.float64), d["R"].astype(np.float64)
    s = float(d["scale"])
    floor = float(-min(neutral[names.index("LeftToeBase")][1], neutral[names.index("LeftFoot")][1]))  # hips above floor
    J = len(names)
    K = len(d["t"])
    gp = np.zeros((K, J, 3))
    gr = np.tile(np.eye(3), (K, J, 1, 1))
    for k in range(K):
        for c, so in TO_SOMA.items():
            if c in cn and so in names:
                gp[k, names.index(so)] = P[k, cn.index(c)] * s
                gr[k, names.index(so)] = R[k, cn.index(c)]
        # joints without a canonical twin (second neck, jaw, eyes, finger / toe ends): rest offset from the
        # parent, carried by the parent's rotation
        mapped = {names.index(v) for v in TO_SOMA.values() if v in names}
        parents = {nm: p for nm, p in sk.bone_order_names_with_parents}
        for jn in names:
            j = names.index(jn)
            if j in mapped or parents[jn] is None:
                continue
            b = names.index(parents[jn])
            gp[k, j] = gp[k, b] + gr[k, b] @ (neutral[j] - neutral[b])
            gr[k, j] = gr[k, b]
    # our floor is y = 0 and Kimodo's too (root y = hip height above the ground): positions carry over directly
    fi = np.clip(np.round(d["t"] * fps).astype(int), 0, n - 1)
    cons = []
    groups = {"RightHand": d["use_r"], "LeftHand": d["use_l"]}
    for jn, use in groups.items():
        idx = np.nonzero(use)[0]
        if len(idx):
            cons.append(EndEffectorConstraintSet(sk, torch.tensor(fi[idx]), torch.tensor(gp[idx], dtype=torch.float32),
                                                 torch.tensor(gr[idx], dtype=torch.float32), None, joint_names=[jn]))
    if "r2_t" in d and len(d["r2_t"]):                     # walking route: root (x, z) waypoints
        from kimodo.constraints import Root2DConstraintSet
        ri = np.clip(np.round(d["r2_t"] * fps).astype(int), 0, n - 1)
        cons.append(Root2DConstraintSet(sk, torch.tensor(ri), torch.tensor(d["r2_xz"] * s, dtype=torch.float32)))
    if "use_full" in d and d["use_full"].any():             # posture keyframes: the whole body as solved
        from kimodo.constraints import FullBodyConstraintSet
        idx = np.nonzero(d["use_full"])[0]
        cons.append(FullBodyConstraintSet(sk, torch.tensor(fi[idx]), torch.tensor(gp[idx], dtype=torch.float32),
                                          torch.tensor(gr[idx], dtype=torch.float32)))
    idx = np.nonzero(d["use_feet"])[0]
    if len(idx):
        cons.append(EndEffectorConstraintSet(sk, torch.tensor(fi[idx]), torch.tensor(gp[idx], dtype=torch.float32),
                                             torch.tensor(gr[idx], dtype=torch.float32), None,
                                             joint_names=["LeftFoot", "RightFoot"]))
    # all constraint tensors on the model device; constraint sets Kimodo re-creates internally (crop_move) build
    # their joint indices on the CPU - keep them with their frame indices
    if not getattr(EndEffectorConstraintSet, "_frt_dev", False):     # patch once per process
        _init = EndEffectorConstraintSet.__init__

        def _init_dev(self, *a_, **k_):
            _init(self, *a_, **k_)
            dev = self.frame_indices.device
            self.pos_indices, self.rot_indices = self.pos_indices.to(dev), self.rot_indices.to(dev)
        EndEffectorConstraintSet.__init__ = _init_dev
        from kimodo.constraints import FullBodyConstraintSet as _F
        _finit = _F.__init__

        def _finit_dev(self, *a_, **k_):
            _finit(self, *a_, **k_)
            for a2 in ("pos_indices", "rot_indices"):
                if hasattr(self, a2):
                    setattr(self, a2, getattr(self, a2).to(self.frame_indices.device))
        _F.__init__ = _finit_dev
        EndEffectorConstraintSet._frt_dev = True
    for c_ in cons:
        c_.to(device)
    torch.manual_seed(int(d["seed"]) if "seed" in d else 0)
    seg = int(round(6.0 * fps))                            # long motions: ~6 s segments, blended by Kimodo
    k_ = max(1, int(np.ceil(n / seg)))
    lens = [n // k_ + (1 if i < n % k_ else 0) for i in range(k_)]
    prompt = str(d["prompt"]) if "prompt" in d else ""                # (a prompt in the text cache, else none)
    res = model([prompt] * k_, lens, constraint_lst=cons, num_denoising_steps=int(d["steps"]) if "steps" in d else 100,
                num_samples=1, multi_prompt=True, post_processing=False, return_numpy=True)
    pj = res["posed_joints"][0]
    names77 = [nm for nm, _ in build_skeleton(77).bone_order_names_with_parents]
    np.savez(out, posed_joints=pj, names77=np.array(names77), fps=fps, floor=floor,
             global_rot_mats=res["global_rot_mats"][0])
    print("ok", pj.shape, "fps", fps, "constraints", [type(c).__name__ for c in cons])


if __name__ == "__main__":
    main()
