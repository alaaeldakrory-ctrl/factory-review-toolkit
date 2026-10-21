"""Safe USD writing. A .usd that Kit has open must never be rewritten in place (Kit reads crate files lazily and
crashes on a half-new file): build into <name>.building.<ext>, then swap it in.

    with safe_stage("C:/scenes/plant.usd") as stage:
        ...                                   # build
    # saved to plant.building.usd, then renamed over plant.usd (or left beside it if the swap fails)
"""
import contextlib
import os

from pxr import Usd


class SwapFailed(RuntimeError):
    pass


def building_path(final):
    root, ext = os.path.splitext(final)
    return root + ".building" + (ext or ".usd")


@contextlib.contextmanager
def safe_stage(final):
    os.makedirs(os.path.dirname(os.path.abspath(final)) or ".", exist_ok=True)
    tmp = building_path(final)
    if os.path.exists(tmp):
        os.remove(tmp)
    # built in memory and exported: CreateNew would clash with a layer of the same name still registered from an
    # earlier build in this process (a Kit session rebuilding the same scene)
    stage = Usd.Stage.CreateInMemory(os.path.basename(final))
    yield stage
    stage.GetRootLayer().Export(tmp)
    del stage
    try:
        os.replace(tmp, final)
    except OSError as e:
        raise SwapFailed(f"could not replace {final} ({e}); is it open in an app? The new build is at {tmp}") from e


def is_locked(path):
    """True if another process holds the file (on Windows a rename fails)."""
    if not os.path.exists(path):
        return False
    probe = path + ".lockcheck"
    try:
        os.rename(path, probe)
    except OSError:
        return True
    os.rename(probe, path)
    return False
