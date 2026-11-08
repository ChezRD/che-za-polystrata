"""serve/winjob.py - tie the processes PolyStrata starts to the server's own lifetime.

Closing the console window, Task Manager or a crash end the server without running its cleanup, and the engine,
the vision encoder and the MCP servers kept running on their own.  On Windows, `contain(proc)` puts a child in a job
object that is set to kill everything in it when its last handle closes - and the only handle is this process's, which
the OS closes however the server ends.  Processes the child starts later join the same job.  The server itself stays
out of the job, so a browser that `--open` starts is not tied to it.  Each child (and the server) is also opted out of
Windows power throttling, see `no_throttle` (#691).

On Linux `spawn()` starts a child in its own session with `PR_SET_PDEATHSIG` = `SIGKILL`, so the kernel ends
it when this process does, even by `SIGKILL` (which runs no cleanup).  The same session is its own process group, so
`signal_group()` can end the child and anything it started.  `contain(proc)` reports whether a Linux child got that
session (it cannot be arranged after the start).  Elsewhere, and if the job cannot be made, a no-op.
"""
from __future__ import annotations

import os
import queue
import signal
import subprocess
import sys
import threading

_job = None
_lock = threading.Lock()

_LINUX = sys.platform.startswith("linux")
_PR_SET_PDEATHSIG = 1

if _LINUX:
    try:                                # loaded here, in the main thread: the forked child must not load anything
        import ctypes

        _LIBC = ctypes.CDLL(None, use_errno=True)
    except (ImportError, OSError):
        _LIBC = None
else:
    _LIBC = None


def _set_pdeathsig() -> None:
    """In the forked child, before exec: SIGKILL when the process that started it ends.  Best effort (a kernel or
    seccomp policy without prctl leaves the child as before)."""
    if _LIBC is not None:
        try:
            _LIBC.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
        except (OSError, AttributeError, ValueError):
            pass


def spawn_kwargs(cpus: list[int] | None = None) -> dict:
    """Popen keyword arguments that tie a child to this process.  Linux: its own session and process group, and
    PR_SET_PDEATHSIG=SIGKILL; `cpus` pins it when given.  Pipes are always close-on-exec and every other file
    descriptor is closed in the child (`close_fds`, the default on POSIX, said here so it is not lost).

    The affinity setting that used to be a `preexec_fn` of its own is folded in: Python allows one, and the parent
    check after prctl closes the fork-to-prctl window in which the server could already have died.

    Linux's parent-death signal follows the *thread* that forked the child, not the whole process: a child started
    from a request or restart thread would be killed when that thread ends.  The server therefore starts its children
    with `spawn()`, on one thread that lives as long as the process."""
    kw = {"close_fds": True}
    if os.name == "nt":
        return kw
    kw["start_new_session"] = True
    parent = os.getpid()

    def setup():
        if cpus and hasattr(os, "sched_setaffinity"):
            try:
                os.sched_setaffinity(0, cpus)
            except OSError:
                pass
        if _LINUX:
            _set_pdeathsig()
            if os.getppid() != parent:          # the server died between fork and prctl
                os._exit(1)

    kw["preexec_fn"] = setup
    return kw


_spawn_queue: "queue.Queue" = queue.Queue()
_spawner = None


def _spawn_loop() -> None:
    """subprocess.Popen for the whole server, on the one thread that is not short-lived (see spawn_kwargs())."""
    while True:
        args, kw, box, done = _spawn_queue.get()
        try:
            box["proc"] = subprocess.Popen(args, **kw)
        except BaseException as e:                  # noqa: BLE001 - re-raised in the caller's thread
            box["error"] = e
        finally:
            done.set()


def spawn(args, **kw):
    """subprocess.Popen(args, **kw) on the server's spawner thread: the parent-death signal is bound to that thread,
    which lives as long as the server, so only the server's own end kills the child.  Exceptions are the caller's."""
    global _spawner
    with _lock:
        if _spawner is None or not _spawner.is_alive():
            _spawner = threading.Thread(target=_spawn_loop, name="child-spawner", daemon=True)
            _spawner.start()
    box, done = {}, threading.Event()
    _spawn_queue.put((list(args), kw, box, done))
    done.wait()
    if "error" in box:
        raise box["error"]
    return box["proc"]


def signal_group(proc, sig) -> bool:
    """Send `sig` to the child and everything it started (POSIX: its process group; Windows: the process itself,
    Python cannot signal a tree there).  True if it was sent."""
    if proc is None:
        return False
    try:
        if os.name != "nt":
            pgid = os.getpgid(proc.pid)
            if pgid == proc.pid:                    # its own session (spawn_kwargs): never the server's own group
                os.killpg(pgid, sig)
                return True
    except (AttributeError, OSError, TypeError, ValueError):
        pass
    try:
        if sig == signal.SIGTERM:
            proc.terminate()
        elif sig == signal.SIGKILL:
            proc.kill()
        else:
            proc.send_signal(sig)
        return True
    except (AttributeError, OSError, ValueError, TypeError):
        return False


if os.name == "nt":
    import ctypes
    from ctypes import wintypes

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class _IoCounters(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in ("ReadOperationCount", "WriteOperationCount",
                                                       "OtherOperationCount", "ReadTransferCount",
                                                       "WriteTransferCount", "OtherTransferCount")]

    class _BasicLimits(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class _ExtendedLimits(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", _BasicLimits), ("IoInfo", _IoCounters),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    _JobObjectExtendedLimitInformation = 9

    _k32.CreateJobObjectW.restype = wintypes.HANDLE
    _k32.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
    _k32.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD)
    _k32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)

    def _make_job():
        job = _k32.CreateJobObjectW(None, None)     # no security attributes: the handle is not inherited
        if not job:
            return None
        info = _ExtendedLimits()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not _k32.SetInformationJobObject(job, _JobObjectExtendedLimitInformation, ctypes.byref(info),
                                            ctypes.sizeof(info)):
            _k32.CloseHandle(job)
            return None
        return job

    class _PowerThrottling(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

    _k32.SetProcessInformation.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD)
    _k32.GetCurrentProcess.restype = wintypes.HANDLE

    def no_throttle(handle) -> bool:
        """Opt a process out of Windows power throttling (EcoQoS).  Without it Windows 11 treats a process whose console
        window is minimized as background work and, on a hybrid CPU, moves its threads to the E-cores: generation
        slowed by about 20% while the server window was minimized (#691, i9-13980HX: 36.6 -> 43.2 tok/s)."""
        for mask in (0x1 | 0x4, 0x1):               # EXECUTION_SPEED and IGNORE_TIMER_RESOLUTION, both off (Windows 11);
            info = _PowerThrottling(1, mask, 0)     # Windows 10 knows the first only: ERROR_INVALID_PARAMETER for both
            if _k32.SetProcessInformation(handle, 4, ctypes.byref(info), ctypes.sizeof(info)):   # ProcessPowerThrottling
                return True
        return False

    no_throttle(_k32.GetCurrentProcess())           # the server itself (tokenizing, streaming) too
else:
    def no_throttle(handle) -> bool:
        return False


def contain(proc) -> bool:
    """Put a started subprocess.Popen in the kill-on-close job. True if it is in; False (harmlessly) otherwise.
    Linux: the child of `spawn_kwargs()` is already in its own session, which the kernel ends with this process
    (PR_SET_PDEATHSIG); True says it has that.  It cannot be arranged after the start, so a child started without
    `spawn_kwargs()` gets False."""
    global _job
    if proc is None:
        return False
    if os.name != "nt":
        try:
            return os.getpgid(proc.pid) == proc.pid
        except (AttributeError, OSError, TypeError, ValueError):
            return False
    with _lock:
        if _job is None:
            _job = _make_job() or 0
        if not _job:
            return False
    try:
        no_throttle(int(proc._handle))
        return bool(_k32.AssignProcessToJobObject(_job, int(proc._handle)))
    except (AttributeError, OSError, ValueError):
        return False
