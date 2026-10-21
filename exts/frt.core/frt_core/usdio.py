"""Safe USD writing. A .usd that Kit has open must never be rewritten in place (Kit reads crate files lazily and
crashes on a half-new file): build into <name>.building.<ext>, then swap it in.

    with safe_stage("C:/scenes/plant.usd") as stage:
        ...                                   # build
    # saved to plant.building.usd, then renamed over plant.usd (or left beside it if the swap fails)

Windows often lets the rename through even while Kit has the file open, so the Kit extension leaves a marker
(<scene>.open, holding its process id) beside every scene it has open, and safe_stage refuses to replace a scene
whose marker belongs to a running process. Close the scene in Kit, or build from the Factory Review window (it
closes the scene first).
"""
import contextlib
import os
import sys

from pxr import Sdf, Usd


class SwapFailed(RuntimeError):
    pass


def building_path(final):
    root, ext = os.path.splitext(final)
    return root + ".building" + (ext or ".usd")


def marker_path(scene):
    return scene + ".open"


def _pid_alive(pid):
    if pid == os.getpid():
        return True
    if sys.platform == "win32":
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, pid)                  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return bool(ok) and code.value == 259                  # STILL_ACTIVE
    try:
        os.kill(pid, 0)                                        # signal 0: existence check (POSIX only)
    except OSError:
        return False
    return True


def mark_open(scene):
    try:
        with open(marker_path(scene), "w") as fh:
            fh.write(str(os.getpid()))
    except OSError:
        pass


def clear_open(scene):
    try:
        os.remove(marker_path(scene))
    except OSError:
        pass


def open_by(scene):
    """Process id of a running app that has `scene` open (from its marker), else None. Stale markers are removed."""
    m = marker_path(scene)
    if not os.path.exists(m):
        return None
    try:
        pid = int(open(m).read().strip() or 0)
    except (OSError, ValueError):
        pid = 0
    if pid and _pid_alive(pid):
        return pid
    clear_open(scene)
    return None


@contextlib.contextmanager
def safe_stage(final, force=False):
    pid = open_by(final)
    if pid and not force:
        raise SwapFailed(f"{final} is open in another app (process {pid}); close it there, or rebuild from the "
                         f"Factory Review window, which closes it first")
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
    try:                                      # generated textures travel with the scene (./textures next to it)
        from . import textures as GT
        GT.localize(final)
    except Exception as e:                    # never lose a build over textures
        print(f"[frt] textures not copied next to {final}: {e}")
    # a layer for this path still cached in this process (the scene was open before) would hand back the old file
    cached = Sdf.Layer.Find(final)
    if cached:
        cached.Reload(True)


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
