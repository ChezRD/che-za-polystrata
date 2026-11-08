"""serve/winjob.py: a contained child - and whatever it starts - ends when the process that contained it ends, even
when that process is killed outright (the console window closed, Task Manager), with no cleanup of its own.

    python -m unittest serve.test_winjob      (the job-object test on Windows, the session/PDEATHSIG test on Linux)
"""
import os
import signal
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# stands in for the server: contains a child that starts a grandchild, prints both pids, then waits to be killed
PARENT = f"""
import subprocess, sys, time
sys.path.insert(0, {str(ROOT)!r})
from serve.winjob import contain
code = ("import subprocess, sys, time; g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); "
        "print(g.pid, flush=True); time.sleep(120)")
child = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
ok = contain(child)
print(ok, child.pid, child.stdout.readline().strip(), flush=True)
time.sleep(120)
"""


def running(pid: int) -> bool:
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
    return str(pid) in out


@unittest.skipUnless(os.name == "nt", "job objects are Windows-only")
class TestWinJob(unittest.TestCase):
    def test_children_end_with_a_killed_server(self):
        # a venv's python.exe is a launcher that runs the real interpreter as its child: kill() must reach the latter
        python = getattr(sys, "_base_executable", None) or sys.executable
        parent = subprocess.Popen([python, "-c", PARENT], stdout=subprocess.PIPE, text=True)
        child = grandchild = None
        try:
            ok, child, grandchild = parent.stdout.readline().split()
            child, grandchild = int(child), int(grandchild)
            self.assertEqual(ok, "True")
            self.assertTrue(running(child) and running(grandchild))
            parent.kill()                       # TerminateProcess: no atexit, no finally - like closing the window
            parent.wait(10)
            deadline = time.monotonic() + 10
            while (running(child) or running(grandchild)) and time.monotonic() < deadline:
                time.sleep(0.2)
            self.assertFalse(running(child), "the child outlived the server")
            self.assertFalse(running(grandchild), "the grandchild outlived the server")
        finally:
            if parent.poll() is None:
                parent.kill()
            for pid in (child, grandchild):              # on a failure, don't leave them sleeping
                if pid and running(pid):
                    subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)

    def test_elsewhere_or_without_a_process_it_is_a_no_op(self):
        from serve.winjob import contain
        self.assertFalse(contain(None))


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(os.name == "nt", "power throttling is a Windows setting")
class TestNoThrottle(unittest.TestCase):
    def test_a_contained_child_is_opted_out_of_power_throttling(self):
        # #691: contain() also sets ProcessPowerThrottling (EcoQoS off), so a minimized server window does not move the
        # engine to the E-cores
        from serve import winjob
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            self.assertTrue(winjob.no_throttle(int(child._handle)))
            self.assertTrue(winjob.contain(child))
        finally:
            child.kill()
            child.wait(10)


# stands in for the server on Linux: starts a grandchild through spawn() (its own long-lived spawner thread),
# prints the grandchild's pid, then sleeps
LINUX_PARENT = f"""
import subprocess, sys, time
sys.path.insert(0, {str(ROOT)!r})
from serve.winjob import spawn, spawn_kwargs
child = spawn([sys.executable, "-c", "import time; time.sleep(120)"], **spawn_kwargs())
print(child.pid, flush=True)
time.sleep(120)
"""


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@unittest.skipUnless(sys.platform.startswith("linux"), "PDEATHSIG and process groups are the Linux mechanism")
class TestLinuxParentDeath(unittest.TestCase):
    def test_the_child_does_not_outlive_a_killed_server(self):
        """0.1: kill -9 of the server leaves no engine behind (the kernel sends SIGKILL to the child)."""
        parent = subprocess.Popen([sys.executable, "-c", LINUX_PARENT], stdout=subprocess.PIPE, text=True)
        child = None
        try:
            child = int(parent.stdout.readline().strip())
            self.assertTrue(alive(child))
            parent.kill()                                   # no atexit, no finally - like the OOM killer or kill -9
            parent.wait(10)
            deadline = time.monotonic() + 5
            while alive(child) and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertFalse(alive(child), "the engine outlived a SIGKILLed server")
        finally:
            if parent.poll() is None:
                parent.kill()
            parent.stdout.close()
            if child and alive(child):
                try:
                    os.killpg(os.getpgid(child), signal.SIGKILL)
                except OSError:
                    pass

    def test_a_child_of_spawn_kwargs_is_its_own_session_and_contained(self):
        from serve.winjob import contain, signal_group, spawn_kwargs
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], **spawn_kwargs())
        try:
            self.assertEqual(os.getpgid(child.pid), child.pid)
            self.assertTrue(contain(child))
        finally:
            self.assertTrue(signal_group(child, signal.SIGKILL))
            child.wait(10)

    def test_a_child_started_without_spawn_kwargs_is_not_contained(self):
        from serve.winjob import contain
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            self.assertNotEqual(os.getpgid(child.pid), child.pid)
            self.assertFalse(contain(child))
        finally:
            child.kill()
            child.wait(10)

    def test_a_child_of_a_short_lived_thread_survives_that_thread(self):
        """The parent-death signal follows the forking thread: spawn() keeps it on the server's own long-lived
        thread, so a restart or a request thread ending must not kill the engine it started."""
        from serve.winjob import signal_group, spawn, spawn_kwargs
        started = {}

        def start():
            started["proc"] = spawn([sys.executable, "-c", "import time; time.sleep(60)"], **spawn_kwargs())

        thread = threading.Thread(target=start)
        thread.start()
        thread.join(10)
        child = started["proc"]
        try:
            time.sleep(0.3)                              # the forking thread is gone by now
            self.assertIsNone(child.poll(), "the engine died with the thread that started it")
        finally:
            signal_group(child, signal.SIGKILL)
            child.wait(10)
