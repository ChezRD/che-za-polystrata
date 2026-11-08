"""Request inspection through HTTP; no native model or GPU is required."""
import contextlib
import io
import json
import threading
import time
import unittest
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import (API_MONITOR_DEFAULTS, ByteTokenizer, EngineDied, MockEngine, Service, api_monitor_settings,
                          serve)


class ReloadableEngine(MockEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.running, self.unloaded = True, False

    def alive(self):
        return self.running

    def unload(self):
        self.running, self.unloaded = False, True

    def restart(self):
        self.running, self.unloaded = True, False


class MonitorOff(unittest.TestCase):
    """#332 is opt-in: by default no prompt or answer is kept, and the monitor's endpoints do not exist."""

    def test_off_by_default(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "Hello.", max_context=4096), tok,
                      ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            req = urllib.request.Request(base + "/v1/chat/completions", headers={"Content-Type": "application/json"},
                                         data=json.dumps({"messages": [{"role": "user", "content": "secret"}]}).encode())
            with urllib.request.urlopen(req, timeout=10) as r:
                self.assertEqual(r.status, 200)
            self.assertEqual(len(svc.api_requests), 0)
            with urllib.request.urlopen(base + "/health", timeout=10) as r:
                self.assertIs(json.load(r)["api_monitor"], False)     # #142: the Requests tab shows how to turn it on
            for path in ("/api/requests", "/api-monitor"):
                with self.assertRaises(urllib.error.HTTPError) as e:
                    urllib.request.urlopen(base + path, timeout=10)
                self.assertEqual(e.exception.code, 404, path)
                e.exception.close()
        finally:
            httpd.shutdown()
            httpd.server_close()


class Monitor(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = ReloadableEngine(self.tok, "Thought.</think>\nHello <script>alert(1)</script>.", max_context=16384)
        self.svc = Service(self.engine, self.tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.svc.api_monitor = True                       # opt-in ("api_monitor": true / --api-monitor)
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def request(self, path, body=None, headers=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            response = urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            raw = response.read().decode()
            return response.status, json.loads(raw) if "application/json" in response.headers.get("Content-Type", "") else raw

    def chat(self, api="openai", **kwargs):
        # Anthropic thinks only when asked (#278): "thinking" there, reasoning_effort for OpenAI
        think = {"thinking": {"type": "enabled", "budget_tokens": 1024}} if api == "anthropic" else             {"reasoning_effort": "low"}
        return self.request("/v1/messages" if api == "anthropic" else "/v1/chat/completions",
                            {"messages": [{"role": "user", "content": "Say hello."}],
                             "max_tokens": 256, **think, **kwargs})

    def latest(self):
        summary = self.request("/api/requests")[1]["requests"][0]
        return self.request("/api/requests?id=" + summary["id"])[1]

    def test_output_and_reasoning_for_both_dialects_and_stream_modes(self):
        for api in ("openai", "anthropic"):
            for stream in (False, True):
                with self.subTest(api=api, stream=stream):
                    code, reply = self.chat(api, stream=stream)
                    self.assertEqual(code, 200)
                    record = self.latest()
                    self.assertEqual(record["state"], "completed")
                    self.assertEqual(record["http_status"], 200)
                    self.assertIn("Say hello.", record["input"])
                    self.assertIn("<script>alert(1)</script>", record["output"])
                    self.assertIn("Thought.", record["reasoning"])
                    self.assertGreater(record["usage"].get("completion_tokens", record["usage"].get("output_tokens", 0)), 0)
                    self.assertGreaterEqual(record["wallclock_s"], record["first_token_s"])
                    self.assertNotIn("_clock", record)
                    if not stream:
                        self.assertEqual(json.loads(record["response"]), reply)
                    else:
                        self.assertNotIn("response", record)

    def test_error_and_engine_death_are_not_successes(self):
        self.assertEqual(self.chat(max_tokens=20000)[0], 400)
        self.assertEqual(self.latest()["state"], "error")
        for api in ("openai", "anthropic"):
            for stream in (False, True):
                with self.subTest(api=api, stream=stream), mock.patch.object(self.engine, "generate", side_effect=EngineDied("stopped")):
                    self.chat(api, stream=stream)
                    record = self.latest()
                    self.assertEqual(record["state"], "error")
                    self.assertIn("stopped", record["error"]["message"])
                    self.assertEqual(record["output"], "")
                    self.assertNotIn("timings", record)

    def test_engine_timings_for_every_api(self):
        """The decode speed is the engine's clock.  Anthropic's and the Responses API's events have no field for it,
        so the requests of Claude Code and Codex showed none."""
        plain = self.engine.generate

        def clocked(ids, max_new, sampling, cancel, embeddings=None):
            n = 0
            try:
                for t in plain(ids, max_new, sampling, cancel, embeddings):
                    n += 1
                    yield t
            finally:
                self.engine.last = {"prompt_tokens": len(ids), "generated": n, "reused": 0, "prompt_ms": 40.0,
                                    "decode_ms": n * 10.0, "finish": "stop"}

        with mock.patch.object(self.engine, "generate", clocked):
            for api in ("openai", "anthropic", "responses"):
                for stream in (False, True):
                    with self.subTest(api=api, stream=stream):
                        if api == "responses":
                            code, _ = self.request("/v1/responses", {"model": "m", "input": "Say hello.",
                                                                     "stream": stream})
                        else:
                            code, _ = self.chat(api, stream=stream)
                        self.assertEqual(code, 200)
                        self.assertEqual(self.latest()["timings"]["predicted_per_second"], 100.0)

    def test_history_is_authorized_and_does_not_capture_headers(self):
        self.svc.api_key = "secret"
        auth = {"Authorization": "Bearer secret"}
        self.assertEqual(self.request("/api/requests")[0], 401)
        self.assertEqual(self.request("/api/requests?id=missing")[0], 401)
        self.assertEqual(len(self.svc.api_requests), 0)
        self.assertEqual(self.request("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}, auth)[0], 200)
        summary = self.request("/api/requests", headers=auth)[1]["requests"][0]
        self.assertNotIn("input", summary)
        self.assertNotIn("output", summary)
        self.assertEqual(self.request("/api/requests?id=" + summary["id"])[0], 401)
        detail = self.request("/api/requests?id=" + summary["id"], headers=auth)[1]
        self.assertNotIn("secret", json.dumps(detail))

    def test_capture_is_bounded_and_eviction_returns_404(self):
        huge = "x" * 262200
        first = self.svc.begin_request("/v1/chat/completions", {"messages": [{"content": huge}]})
        self.assertEqual(len(first["input"]), 262144)
        self.assertTrue(first["input_truncated"])
        for _ in range(100):
            self.svc.begin_request("/v1/chat/completions", {"messages": []})
        self.assertEqual(len(self.request("/api/requests")[1]["requests"]), 100)
        self.assertEqual(self.request("/api/requests?id=" + first["id"])[0], 404)
        # Exercise output capture without making a mock engine emit 262K tokens.
        handler = object.__new__(self.httpd.RequestHandlerClass)
        handler.record = self.svc.api_requests[-1]
        chunk = {"choices": [{"delta": {"content": huge, "reasoning_content": huge}}]}
        list(handler._capture((item for item in [chunk]), "openai"))
        for field in ("output", "reasoning"):
            self.assertEqual(len(handler.record[field]), 262144)
            self.assertTrue(handler.record[field + "_truncated"])

    def test_queue_wait_and_loading_are_attributed_separately(self):
        self.engine.unload()
        entered, release = threading.Event(), threading.Event()
        original = self.engine.restart
        def restart():
            entered.set()
            self.assertTrue(release.wait(5))
            original()
        with mock.patch.object(self.engine, "restart", side_effect=restart), ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(self.chat)
            self.assertTrue(entered.wait(5))
            second = pool.submit(self.chat)
            deadline = time.monotonic() + 5
            while len(self.svc.api_requests) < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(len(self.svc.api_requests), 2)
            history = self.request("/api/requests")[1]
            self.assertEqual({r["state"] for r in history["requests"]}, {"queued", "loading"})
            self.assertFalse(history["loaded"])
            self.assertTrue(history["auto_load"])
            time.sleep(0.03)
            release.set()
            self.assertEqual(first.result()[0], 200)
            self.assertEqual(second.result()[0], 200)
        records = list(self.svc.api_requests)
        self.assertGreater(records[0]["load_s"], 0.02)
        self.assertGreater(records[1]["queue_s"], 0.02)
        self.assertLess(records[1]["load_s"], records[0]["load_s"])
        self.assertTrue(all(r["state"] == "completed" for r in records))

    def test_page_and_relative_assets_are_served(self):
        self.assertIs(self.request("/health")[1]["api_monitor"], True)   # #142: the Requests tab embeds the page
        code, html = self.request("/api-monitor")
        self.assertEqual(code, 200)
        self.assertIn('src="/web/monitor.js"', html)
        code, script = self.request("/web/monitor.js")
        self.assertEqual(code, 200)
        self.assertIn("textContent", script)
        self.assertNotIn("innerHTML", script)
        self.assertEqual(self.request("/web/../server.py")[0], 404)


class ConversationCacheCard(unittest.TestCase):
    """#596: /metrics' "conversation_cache": the parked conversations from the engine's log lines, the budget and
    slots from its INFO line, and how much of the prompts the cache gave back."""

    PARK = ("strata serve: conversation cache: parked 5000 tokens in 12.0 ms; parked=1 bytes=104857600 evictions=0 "
            "snapshot_bytes=104857600 reused_kv_bytes=0\n")
    RESTORE = "strata serve: conversation cache: restored 4000 tokens (exact) in 30.0 ms; parked=2 bytes=209715200\n"

    def test_the_log(self):
        from serve.server import ConvCacheLog
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "strata.log"
            log.write_text("an earlier run\n" + self.PARK, encoding="utf-8")
            start = log.stat().st_size                 # this run starts here: the earlier run's park is not counted
            c = ConvCacheLog()
            self.assertEqual(c.poll(str(log), start)["parked"], 0)
            with open(log, "a", encoding="utf-8") as f:
                f.write("strata serve: something else\n" + self.PARK + self.RESTORE + "strata serve: conversation ca")
            st = c.poll(str(log), start)
            self.assertEqual((st["parked"], st["bytes"], st["parks"], st["restores"]), (2, 209715200, 1, 1))
            self.assertEqual((st["last_event"], st["last_tokens"]), ("restored", 4000))
            with open(log, "a", encoding="utf-8") as f:   # the cut line is read once it is whole
                f.write("che: dropped 1 superseded copy of this conversation; parked=1\n")
            self.assertEqual(c.poll(str(log), start)["parked"], 1)
            self.assertEqual(c.poll(str(log), log.stat().st_size)["parked"], 0)   # the engine started again
            self.assertEqual(c.poll(None, None)["parked"], 0)

    def test_metrics(self):
        tok = ByteTokenizer()
        engine = MockEngine(tok, "Thought.</think>\nHello.", max_context=4096)
        engine.info = {"conversation_cache_mib": 8192, "conversation_cache_slots": 4}
        svc = Service(engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
                c = json.loads(r.read())["conversation_cache"]
            self.assertEqual((c["enabled"], c["budget_mib"], c["slots"], c["parked"], c["requests"]),
                             (True, 8192, 4, 0, 0))
            req = urllib.request.Request(base + "/v1/chat/completions", headers={"Content-Type": "application/json"},
                                         data=json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode())
            with urllib.request.urlopen(req, timeout=10) as r:
                r.read()
            with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
                c = json.loads(r.read())["conversation_cache"]
            self.assertEqual(c["requests"], 1)
            self.assertGreater(c["last_prompt"], 0)
            engine.info = {}
            with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
                self.assertFalse(json.loads(r.read())["conversation_cache"]["enabled"])   # off: the reuse lines only
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_the_card_is_on_the_page(self):
        web = Path(__file__).parent / "web"
        html, js = (web / "index.html").read_text(encoding="utf-8"), (web / "app.js").read_text(encoding="utf-8")
        for el in ("cc-card", "cc-slots-text", "cc-mem-text", "cc-facts", "cc-note"):
            self.assertIn(f'id="{el}"', html)
            self.assertIn(f'"{el}"', js)


class Retention(unittest.TestCase):
    """The monitor's bounds: how many requests, how many characters of each field, and about how many bytes in all."""

    def service(self, **limits):
        tok = ByteTokenizer()
        svc = Service(ReloadableEngine(tok, "Hello.", max_context=16384), tok,
                      ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        svc.api_monitor = True
        for key, value in limits.items():
            setattr(svc, f"api_monitor_{key}", value)
        svc.api_requests = __import__("collections").deque(maxlen=svc.api_monitor_max_requests)
        svc.api_requests_bytes = 0
        return svc

    def test_the_deque_follows_max_requests(self):
        svc = self.service(max_requests=3)
        for i in range(5):
            record = svc.begin_request("/v1/chat/completions", {"messages": [{"role": "user", "content": str(i)}]})
            with svc.status_lock:
                svc.api_requests[-1]["state"] = "completed"
        self.assertEqual(len(svc.api_requests), 3)
        self.assertNotIn(record, list(svc.api_requests)[:2])
        self.assertIs(svc.api_requests[-1], record)
        # a new limit at runtime: the deque takes it, keeping the newest records
        svc.api_monitor_max_requests = 2
        svc.begin_request("/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}]})
        self.assertEqual(len(svc.api_requests), 2)
        self.assertEqual(svc.api_requests.maxlen, 2)

    def test_input_is_truncated_to_max_chars(self):
        svc = self.service(max_chars=128)
        record = svc.begin_request("/v1/chat/completions",
                                   {"messages": [{"role": "user", "content": "x" * 5000}]})
        self.assertEqual(len(record["input"]), 128)
        self.assertTrue(record["input_truncated"])

    def test_the_byte_cap_evicts_the_oldest_and_keeps_the_newest(self):
        svc = self.service(max_requests=10, max_chars=2000, max_bytes=4096, keep_reasoning=False)
        first = svc.begin_request("/v1/chat/completions", {"messages": [{"role": "user", "content": "a" * 2000}]})
        second = svc.begin_request("/v1/chat/completions", {"messages": [{"role": "user", "content": "b" * 2000}]})
        self.assertNotIn(first, list(svc.api_requests))
        self.assertIs(svc.api_requests[-1], second)
        self.assertLessEqual(svc.api_requests_bytes, svc.api_monitor_max_bytes)
        # one record over the cap alone is trimmed instead of thrown away
        svc.api_monitor_max_bytes = 1024
        third = svc.begin_request("/v1/chat/completions", {"messages": [{"role": "user", "content": "c" * 5000}]})
        self.assertIs(svc.api_requests[-1], third)
        self.assertLessEqual(svc.api_requests_bytes, 1024)
        self.assertLessEqual(len(third["input"]), 1024)
        self.assertTrue(third["input_truncated"])

    def test_append_counts_only_the_bytes_that_stay(self):
        svc = self.service(max_chars=4, max_bytes=0)
        record = svc.begin_request("/v1/chat/completions", {"messages": [{"role": "user", "content": "go"}]})
        with svc.status_lock:
            for piece in ("я", "я", "я", "a", "a", "a"):
                svc._append_text(record, "output", piece, svc.api_monitor_max_chars)
        self.assertEqual(record["output"], "яяяa")
        self.assertTrue(record["output_truncated"])
        self.assertEqual(svc.api_requests_bytes, sum(svc._record_bytes(r) for r in svc.api_requests))

    def test_trim_cuts_utf8_bytes_on_a_character_boundary(self):
        record = {"input": "я" * 100, "output": "", "reasoning": "", "response": ""}
        Service._trim_record(record, 400)
        self.assertEqual(record["input"], "я" * 72)
        self.assertEqual(Service._record_bytes(record), 400)
        self.assertTrue(record["input_truncated"])
        mid = {"input": "я" * 10}
        Service._trim_record(mid, 269)
        self.assertEqual(mid["input"], "я" * 6)
        self.assertLessEqual(Service._record_bytes(mid), 269)

    def test_streamed_mixed_width_text_matches_the_byte_counter(self):
        svc = self.service(max_chars=4, max_bytes=0)
        svc.engine.script = svc.tok.encode("яяя" + "a" * 30)
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            body = json.dumps({"messages": [{"role": "user", "content": "go"}], "max_tokens": 80,
                               "stream": True, "reasoning_effort": "none"}).encode()
            req = urllib.request.Request(base + "/v1/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                self.assertEqual(r.status, 200)
                r.read()
        finally:
            httpd.shutdown()
            httpd.server_close()
        record = svc.api_requests[-1]
        self.assertEqual(record["output"], "яяяa")
        self.assertTrue(record["output_truncated"])
        self.assertEqual(svc.api_requests_bytes, sum(svc._record_bytes(rec) for rec in svc.api_requests))

    def test_reasoning_can_be_left_out(self):
        svc = self.service(keep_reasoning=False)
        self.assertIs(svc.api_monitor_keep_reasoning, False)
        self.assertEqual(svc.api_monitor_limits()["keep_reasoning"], False)
        svc.engine.script = svc.tok.encode("Thought.</think>\nHello.")   # the thinking comes, the answer follows
        # the HTTP path: a reply with thinking, kept as the output alone
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            body = json.dumps({"messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 64,
                               "reasoning_effort": "low"}).encode()
            req = urllib.request.Request(base + "/v1/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as r:
                r.read()
            with urllib.request.urlopen(base + "/api/requests", timeout=10) as r:
                listed = json.load(r)
        finally:
            httpd.shutdown()
            httpd.server_close()
        record = svc.request_records(listed["requests"][0]["id"])
        self.assertEqual(record["reasoning"], "")
        self.assertFalse(record["reasoning_truncated"])
        self.assertIn("Hello.", record["output"])
        self.assertEqual(listed["retention"], svc.api_monitor_max_requests)
        self.assertEqual(listed["limits"], svc.api_monitor_limits())

    def test_output_is_truncated_to_max_chars(self):
        svc = self.service(max_chars=512)
        svc.engine.script = svc.tok.encode("y" * 4000)
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            body = json.dumps({"messages": [{"role": "user", "content": "go"}], "max_tokens": 4000}).encode()
            req = urllib.request.Request(base + "/v1/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                r.read()
        finally:
            httpd.shutdown()
            httpd.server_close()
        record = svc.request_records(svc.api_requests[-1]["id"])
        self.assertLessEqual(len(record["output"]), 512)
        self.assertLessEqual(len(record["response"]), 512)
        self.assertTrue(record["output_truncated"] or record["response_truncated"])

    def test_health_reports_the_limits(self):
        svc = self.service(max_requests=7, max_chars=2048, max_bytes=0, keep_reasoning=False)
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            with urllib.request.urlopen(base + "/health", timeout=10) as r:
                health = json.load(r)
        finally:
            httpd.shutdown()
            httpd.server_close()
        self.assertTrue(health["api_monitor"])
        self.assertEqual(health["limits"], {"max_requests": 7, "max_chars": 2048, "max_bytes": 0,
                                            "keep_reasoning": False})


class Config(unittest.TestCase):
    def settings(self, cfg, forced=False):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            enabled, limits = api_monitor_settings(cfg, forced)
        return enabled, limits, out.getvalue()

    def test_off_and_the_old_true(self):
        self.assertEqual(self.settings({})[:2], (False, API_MONITOR_DEFAULTS))
        self.assertEqual(self.settings({"api_monitor": False})[:2], (False, API_MONITOR_DEFAULTS))
        self.assertTrue(self.settings({"api_monitor": True})[0])
        self.assertTrue(self.settings({}, forced=True)[0])

    def test_the_object_form(self):
        enabled, limits, _ = self.settings({"api_monitor": {"max_requests": 50, "max_chars": 65536,
                                                            "max_bytes": 33554432, "keep_reasoning": False}})
        self.assertTrue(enabled)
        self.assertEqual(limits, {"max_requests": 50, "max_chars": 65536, "max_bytes": 33554432,
                                  "keep_reasoning": False})

    def test_object_keys_win_over_flat_keys(self):
        _, limits, _ = self.settings({"api_monitor": {"max_chars": 4096}, "api_monitor_max_chars": 8192,
                                      "api_monitor_max_requests": 9})
        self.assertEqual((limits["max_chars"], limits["max_requests"]), (4096, 9))

    def test_flat_keys_alone(self):
        enabled, limits, _ = self.settings({"api_monitor": True, "api_monitor_max_requests": 5,
                                            "api_monitor_max_bytes": 0})
        self.assertTrue(enabled)
        self.assertEqual((limits["max_requests"], limits["max_bytes"]), (5, 0))

    def test_clamps_and_warnings(self):
        _, limits, out = self.settings({"api_monitor": {"max_requests": 5000, "max_chars": 10,
                                                        "max_bytes": 1 << 40, "keep_reasoning": "no"}})
        self.assertEqual((limits["max_requests"], limits["max_chars"], limits["max_bytes"], limits["keep_reasoning"]),
                         (1000, 1024, 1 << 30, True))
        self.assertIn("out of range", out)
        self.assertIn("expected true or false", out)
        _, limits, out = self.settings({"api_monitor": {"max_requests": "many"}})
        self.assertEqual(limits["max_requests"], 100)
        self.assertIn("expected a whole number", out)

    def test_an_invalid_form_says_so(self):
        enabled, limits, out = self.settings({"api_monitor": "yes"})
        self.assertFalse(enabled)
        self.assertEqual(limits, API_MONITOR_DEFAULTS)
        self.assertIn("expected true, false or an object", out)


if __name__ == "__main__":
    unittest.main()
