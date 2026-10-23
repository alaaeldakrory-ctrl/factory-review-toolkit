"""Run NVIDIA Kimodo (https://github.com/nv-tlabs/kimodo, Apache-2.0 code; Kimodo-SOMA-RP models under the NVIDIA
Open Model License) from its source checkout, in its own Python environment, without the gated Llama text encoder:
a stand-in encoder returns "no text" (Kimodo treats an empty prompt as zero text features), so motion is driven by
the constraints (end-effector / full-body keyframes, root path) and the model's learned motion prior.

Runs with the Kimodo env's python (not the toolkit's):
    C:/OmniverseData/kimodo/env/Scripts/python.exe tools/kimodo/kimodo_run.py job.json out.npz
job.json: {"seconds": 5.0, "constraints": [...Kimodo constraint dicts...], "seed": 0, "steps": 100,
           "prompt": ""}
out.npz: Kimodo's output arrays (posed_joints, global_rot_mats, local_rot_mats, root_positions, foot_contacts...),
plus "joint_names" and "fps".
"""
import json
import os
import sys

KIMODO_SRC = os.environ.get("KIMODO_SRC", r"C:/OmniverseData/kimodo/src")
sys.path.insert(0, KIMODO_SRC)
os.environ.setdefault("TEXT_ENCODER_DEVICE", "cpu")

import numpy as np   # noqa: E402
import torch   # noqa: E402


class NoText:
    """Stand-in text encoder: zero features (Kimodo's own handling of an empty prompt)."""

    def __init__(self, dim, device):
        self.dim, self.device = dim, device

    def __call__(self, texts):
        n = len(texts)
        return torch.zeros((n, 1, self.dim), device=self.device), [0] * n

    def to(self, *a, **k):
        return self


def text_dim(model_path):
    from omegaconf import OmegaConf
    conf = OmegaConf.load(os.path.join(model_path, "config.yaml"))
    for key in ("text_dim", "text_feat_dim", "llm_dim"):
        v = OmegaConf.select(conf, key, default=None) or OmegaConf.select(conf, f"denoiser.{key}", default=None)
        if v:
            return int(v)
    txt = OmegaConf.to_yaml(conf)
    for ln in txt.splitlines():
        if "text" in ln and "dim" in ln and ":" in ln:
            try:
                return int(ln.split(":")[1])
            except ValueError:
                pass
    return 4096


def main():
    job = json.load(open(sys.argv[1]))
    out = sys.argv[2]
    from kimodo import load_model
    from kimodo.constraints import load_constraints_lst
    from kimodo.model.load_model import _resolve_hf_model_path
    name = job.get("model", "kimodo-soma-rp-v1.2")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    path = _resolve_hf_model_path(name)
    dim = int(job.get("text_dim") or text_dim(str(path)))
    model = load_model(name, device=device, text_encoder=NoText(dim, device))
    fps = float(model.fps)
    n = int(round(float(job.get("seconds", 5.0)) * fps))
    cons = []
    if job.get("constraints"):
        tmp = out + ".constraints.json"
        json.dump(job["constraints"], open(tmp, "w"))
        cons = load_constraints_lst(tmp, model.skeleton)
    torch.manual_seed(int(job.get("seed", 0)))
    np.random.seed(int(job.get("seed", 0)))
    kw = {}
    if job.get("first_heading") is not None:
        kw["first_heading_angle"] = torch.tensor([float(job["first_heading"])], device=device)
    res = model([job.get("prompt", "")], [n], constraint_lst=cons, num_denoising_steps=int(job.get("steps", 100)),
                num_samples=1, multi_prompt=True, post_processing=False, return_numpy=True, **kw)
    names = list(getattr(model.skeleton, "bone_order_names", None) or getattr(model.skeleton, "joint_names", []) or [])
    arrays = {k: (v[0] if hasattr(v, "shape") and v.shape and v.shape[0] == 1 else v) for k, v in res.items()
              if hasattr(v, "shape")}
    np.savez(out, joint_names=np.array(names), fps=fps, **arrays)
    print("ok", {k: tuple(v.shape) for k, v in arrays.items()}, "fps", fps, "joints", len(names))


if __name__ == "__main__":
    main()
