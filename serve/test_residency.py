"""Phase 0 of the single-GPU residency work: the safety around a model switch.

0.2 the new model's VRAM is checked before the old one is evicted; 0.3 the release is confirmed (processes gone and
free VRAM back) before a load starts; 0.4 a failed start gets a cooldown, is retried by nobody for a while, leaves
503 + Retry-After, and the earlier model is loaded again on a background thread.  The engine is a stand-in: no GPU
is needed, and no real model is started.

    python -m unittest serve.test_residency -v
"""
import json
import math
import os
import sys
import tempfile
import threading
import time
import types
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from serve import models as registry
from serve import server as S
from serve.frontend import ChatTemplate
from serve.server import (ByteTokenizer, EngineDied, MockEngine, Service, StrataEngine, SwitchFailed, config_key,
                          serve)
from serve.test_models import Stopped, install


def until(cond, t=10.0):
    end = time.time() + t
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


class SwitchBase(unittest.TestCase):
    """A registry of fake models (serve.test_models.install) and a service whose old engine is a counter."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)
        self.a = install(self.root, "alpha", "qwen35moe", "alpha.gguf")
        self.b = install(self.root, "beta", "gemma4", "beta.gguf")
        self.g = install(self.root, "gamma", "gemma4", "gamma.gguf")
        registry._seen.clear()
        self.tok = ByteTokenizer()
        self.old = Stopped(self.tok)
        self.svc = Service(self.old, self.tok, ChatTemplate(S.ROOT / "serve/chat_template.jinja"), model_name="alpha")
        self.svc.config_path, self.svc.models_root = str(self.a), self.root
        self.started = []
        self.attempts = 0
        self.fails = set()
        self.active = 0
        self.most_active = 0
        self.lock = threading.Lock()

        def start(cfg, tok, lazy=False):
            with self.lock:
                self.active += 1
                self.attempts += 1
                self.most_active = max(self.most_active, self.active)
            try:
                time.sleep(0.15)                      # room for an overlapping switch, if one were possible
                if cfg["model_name"] in self.fails:
                    raise RuntimeError("no VRAM")
                self.started.append(cfg)
                return Stopped(tok, "Hi."), None, None, {}, None, []
            finally:
                with self.lock:
                    self.active -= 1

        for p in (mock.patch.object(S, "load_tokenizer", lambda p: self.tok),
                  mock.patch.object(S, "start_model", start)):
            p.start()
            self.addCleanup(p.stop)

    def configure(self, cfg_path, **keys):
        cfg = json.loads(cfg_path.read_text())
        cfg.update(keys)
        cfg_path.write_text(json.dumps({k: v for k, v in cfg.items() if v is not None}))


class Preflight(SwitchBase):
    def test_a_model_that_cannot_fit_leaves_the_old_one_running(self):
        """0.2: the estimate is checked before anything is stopped."""
        self.configure(self.b, vram_need_mib=12000)
        with mock.patch.object(self.svc, "free_vram_mib", return_value=500), \
                mock.patch.object(self.svc, "gpu_process_mib", return_value={}):
            with self.assertRaises(SwitchFailed) as caught:
                self.svc.switch_model("beta")
        self.assertIn("needs 12000 MiB", str(caught.exception))
        self.assertIn("not unloaded", str(caught.exception))
        self.assertTrue(caught.exception.retry_after)
        self.assertEqual((self.old.stops, self.old.running, self.svc.model), (0, True, "alpha"))
        self.assertEqual(self.started, [])

    def test_the_vram_the_old_model_gives_back_counts(self):
        """The old process's own usage (per PID) is what makes the new model fit: 500 free + 12000 returned."""
        self.configure(self.b, vram_need_mib=12000)
        self.old.proc = types.SimpleNamespace(pid=4242)
        self.svc.backend = "cuda"

        def processes():
            return {4242: 12000} if self.old.running else {}

        with mock.patch.object(self.svc, "gpu_process_mib", side_effect=processes), \
                mock.patch.object(self.svc, "free_vram_mib", side_effect=lambda: 500 if self.old.running else 13000):
            self.assertEqual(self.svc.switch_model("beta")["current"], True)
        self.assertEqual(self.old.stops, 1)
        self.assertEqual([c["model_name"] for c in self.started], ["beta"])

    def test_an_estimate_alone_does_not_refuse(self):
        """A model's dense floor is a lower bound: with no per-PID reading, the old behaviour (try it) stays."""
        self.configure(self.b, vram_need_mib=None)
        with mock.patch.object(self.svc, "free_vram_mib", return_value=100), \
                mock.patch.object(self.svc, "gpu_process_mib", return_value=None):
            self.svc.switch_model("beta")
        self.assertEqual(self.svc.model, "beta")


class FailedStart(SwitchBase):
    def test_the_old_model_comes_back_on_a_background_thread(self):
        self.fails.add("beta")
        with self.assertRaises(SwitchFailed) as caught:
            self.svc.switch_model("beta")
        self.assertIn("loaded again in the background", str(caught.exception))
        self.assertGreater(caught.exception.retry_after or 0, 0)
        self.assertTrue(until(lambda: self.old.running), "the earlier model was not started again")
        self.assertTrue(until(lambda: self.svc.last_good == config_key(self.a)))
        self.assertEqual((self.old.stops, self.old.starts), (1, 1))
        self.assertEqual(self.svc.model, "alpha")

    def test_a_broken_model_is_not_retried_by_every_request(self):
        """0.4: the failure gets a cooldown; the next switch to it is refused before anything is touched."""
        self.fails.add("beta")
        with self.assertRaises(SwitchFailed):
            self.svc.switch_model("beta")
        self.assertTrue(until(lambda: self.old.running))
        attempts = self.attempts
        with self.assertRaises(SwitchFailed) as caught:
            self.svc.switch_model("beta")
        self.assertIn("not tried again", str(caught.exception))
        self.assertTrue(0 < (caught.exception.retry_after or 0) <= math.ceil(self.svc.switch_fail_cooldown_s))
        self.assertEqual(self.attempts, attempts)              # the failure was not tried again
        self.assertEqual(self.old.stops, 1)                    # and the restored model was not unloaded
        stat = self.svc.model_stats[config_key(self.b)]
        stat["cooldown_until"] = 0                             # the cooldown is over: one request tries again
        with self.assertRaises(SwitchFailed):
            self.svc.switch_model("beta")
        self.assertEqual(self.attempts, attempts + 1)

    def test_a_switch_that_does_not_release_is_not_loaded_into(self):
        """0.3: the free VRAM not rising means the load never starts; the old model is restored instead."""
        self.configure(self.b, vram_need_mib=5000)
        self.old.proc = types.SimpleNamespace(pid=4242)
        with mock.patch.object(self.svc, "gpu_process_mib", return_value={4242: 5000}), \
                mock.patch.object(self.svc, "free_vram_mib", return_value=500), \
                mock.patch.object(self.svc, "release_wait_s", 0.3):
            with self.assertRaises(SwitchFailed) as caught:
                self.svc.switch_model("beta")
        self.assertIn("did not release its VRAM", str(caught.exception))
        self.assertEqual(self.started, [])                     # the new model was never started
        self.assertTrue(until(lambda: self.old.running))


class Serialized(SwitchBase):
    def test_two_switches_do_not_run_together(self):
        """One switch at a time (the gate's job): a second waits for the first to finish."""
        errors = []

        def switch(name):
            try:
                self.svc.switch_model(name)
            except Exception as e:  # noqa: BLE001 - reported below
                errors.append(e)

        threads = [threading.Thread(target=switch, args=(name,)) for name in ("beta", "gamma")]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        self.assertEqual(errors, [])
        self.assertEqual(self.most_active, 1)
        self.assertEqual(sorted(c["model_name"] for c in self.started), ["beta", "gamma"])


class Release(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.svc = Service(MockEngine(self.tok, "ok"), self.tok,
                           ChatTemplate(S.ROOT / "serve/chat_template.jinja"))
        self.svc.release_wait_s = 0.3

    def test_waits_for_the_pids_and_the_free_vram(self):
        held = {"pids": {7: 900}}
        state = {"free": 1000}
        with mock.patch.object(self.svc, "gpu_process_mib", side_effect=lambda: dict(held["pids"])), \
                mock.patch.object(self.svc, "free_vram_mib", side_effect=lambda: state["free"]):
            t0 = time.monotonic()
            self.assertFalse(self.svc.wait_for_release([7], need_mib=2000, free_before=1000, expected_mib=1000))
            self.assertLess(time.monotonic() - t0, 2)          # short deadline, not a minute
            held["pids"], state["free"] = {}, 2100
            self.assertTrue(self.svc.wait_for_release([7], need_mib=2000, free_before=1000, expected_mib=1000))

    def test_no_reader_keeps_the_old_behaviour(self):
        with mock.patch.object(self.svc, "gpu_process_mib", return_value=None), \
                mock.patch.object(self.svc, "free_vram_mib", return_value=None):
            self.assertTrue(self.svc.wait_for_release([7], need_mib=2000))

    def test_nothing_to_wait_for(self):
        self.assertTrue(self.svc.wait_for_release([], None))


class ReadyTimeout(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)

    def script(self, code):
        py = self.root / "engine.py"
        py.write_text("#!" + sys.executable + "\n" + code)
        py.chmod(0o700)
        return str(py)

    def test_the_models_own_ready_timeout(self):
        t0 = time.monotonic()
        with self.assertRaises(EngineDied) as caught:
            StrataEngine(self.script("import time\ntime.sleep(3600)\n"), [], ready_s=0.4)
        self.assertIn("did not report READY within 0.4 s", str(caught.exception))
        self.assertLess(time.monotonic() - t0, 10)

    def test_the_config_numbers(self):
        with mock.patch.object(S, "ENGINE_READY_S", 900.0):
            self.assertEqual(S.model_ready_timeout({}), 900.0)
            self.assertEqual(S.model_ready_timeout({"ready_timeout_s": 12}), 12.0)
            self.assertEqual(S.model_ready_timeout({"ready_timeout_s": 0}), math.inf)


class ReadyTimeoutInASwitch(SwitchBase):
    def test_the_switch_uses_the_models_timeout_and_defaults_to_300(self):
        self.configure(self.b, ready_timeout_s=12)
        self.svc.switch_model("beta")
        self.assertEqual(self.started[-1]["ready_timeout_s"], 12)
        self.configure(self.g, ready_timeout_s=None)
        self.svc.switch_model("gamma")
        self.assertEqual(self.started[-1]["ready_timeout_s"], 300)
        self.assertEqual(self.svc.ready_timeout_s, 300)


class RetryAfter(SwitchBase):
    """The 503 of a failed switch carries Retry-After (over the real HTTP API too)."""

    def setUp(self):
        super().setUp()
        self.svc.switch_on_request = True
        self.httpd = serve(self.svc, port=0)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def request(self, path, body=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, dict(r.headers), json.load(r)
        except urllib.error.HTTPError as e:
            with e:
                return e.code, dict(e.headers), json.load(e)

    def test_a_chat_naming_a_broken_model_is_503_with_retry_after(self):
        self.fails.add("beta")
        code, headers, body = self.request("/v1/chat/completions",
                                           {"model": "beta", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(code, 503)
        self.assertIn("loaded again in the background", body["error"]["message"])
        self.assertTrue(int(headers["Retry-After"]) > 0)

    def test_the_models_page_gets_it_too(self):
        self.fails.add("beta")
        code, headers, _ = self.request("/api/models/load", {"model": "beta"})
        self.assertEqual(code, 503)
        self.assertTrue(int(headers["Retry-After"]) > 0)

    def test_a_busy_admission_is_503_with_retry_after(self):
        """0.4: a switch that cannot get in now is retryable, not a permanent 409."""
        self.svc.switch_wait_s = 0.2
        with self.svc.fifo:
            code, headers, _ = self.request("/v1/chat/completions",
                                            {"model": "beta", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(code, 503)
        self.assertTrue(int(headers["Retry-After"]) > 0)


class CloseWaits(unittest.TestCase):
    """0.1/0.3: QUIT, then TERM, then KILL, with short waits - and the child does not outlive it."""

    @unittest.skipIf(os.name == "nt", "the POSIX process group and SIGTERM are the point here")
    def test_an_engine_that_ignores_quit_and_term_is_killed(self):
        code = ("import signal, time\n"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                "print('READY 4096', flush=True)\n"
                "time.sleep(60)\n")
        with tempfile.TemporaryDirectory() as d:
            py = Path(d) / "stubborn.py"
            py.write_text("#!" + sys.executable + "\n" + code)
            py.chmod(0o700)
            with mock.patch.object(StrataEngine, "QUIT_TIMEOUT_S", 0.2), \
                    mock.patch.object(StrataEngine, "TERM_TIMEOUT_S", 0.3), \
                    mock.patch.object(StrataEngine, "KILL_TIMEOUT_S", 2.0):
                engine = StrataEngine(str(py), [])
                pid = engine.proc.pid
                t0 = time.monotonic()
                engine.close()
                took = time.monotonic() - t0
                self.assertIsNone(engine.proc)
                self.assertGreaterEqual(took, 0.4)               # it did wait out QUIT and TERM
                self.assertLess(took, 5)                         # not the old 60 s
                deadline = time.monotonic() + 5
                while _running(pid) and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertFalse(_running(pid))


def _running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


if __name__ == "__main__":
    unittest.main()
