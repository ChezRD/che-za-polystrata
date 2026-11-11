"""tools/test_embd_extract.py - the copy keeps the tensor's shape, type and bytes; run python3 tools/test_embd_extract.py."""
import contextlib
import io
import pathlib
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import embd_extract
import gguf_reader as G

ALIGN = 32
F32, IQ4_XS, BF16 = 0, 23, 30


def gstr(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def kv_str(key: str, value: str) -> bytes:
    return gstr(key) + struct.pack("<I", 8) + gstr(value)


def kv_u16(key: str, value: int) -> bytes:
    return gstr(key) + struct.pack("<IH", 2, value)


def write_gguf(path: pathlib.Path, tensors, kvs=(), tail: bytes = b"") -> None:
    """GGUF v3 from [(name, shape, type_id, payload)]; each payload starts at its own 32-aligned offset."""
    infos = blobs = b""
    for name, shape, type_id, payload in tensors:
        infos += gstr(name) + struct.pack("<I", len(shape)) + struct.pack("<%dQ" % len(shape), *shape)
        infos += struct.pack("<IQ", type_id, len(blobs))
        blobs += payload + b"\0" * ((-len(payload)) % ALIGN)
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs)) + b"".join(kvs) + infos
    path.write_bytes(head + b"\0" * ((-len(head)) % ALIGN) + blobs + tail)


def pattern(n: int) -> bytes:
    return bytes((i * 37 + 11) % 256 for i in range(n))


class EmbdExtractTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = pathlib.Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def run_tool(self, src: pathlib.Path, out: pathlib.Path):
        argv = ["embd_extract.py", "--gguf", str(src), "--out", str(out)]
        log = io.StringIO()
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(log):
            rc = embd_extract.main()
        return rc, log.getvalue()

    def check_output(self, out: pathlib.Path, shape, type_id: int, payload: bytes, source_name: str) -> None:
        g = G.GGUFFile(out)
        self.assertEqual(len(g.tensors), 1, "the output must hold exactly one tensor")
        t = g.tensors[0]
        self.assertEqual(t.name, "token_embd.weight")
        self.assertEqual(t.shape, shape, "output shape differs from the source")
        self.assertEqual(t.type_id, type_id, "output type id differs from the source")
        self.assertEqual(t.offset, 0, "the one tensor must start at the data section")
        self.assertEqual(g.metadata.get("general.architecture"), "strata-embd")
        self.assertEqual(g.metadata.get("strata.embd.source"), source_name)
        body = out.read_bytes()[g.data_start + t.offset:]
        self.assertEqual(body, payload, "payload bytes (or bytes after them) differ from the source")

    def test_bf16(self):
        payload = pattern(32 * 4 * 2)
        src = self.root / "model-00001-of-00002.gguf"
        write_gguf(src, [("token_embd.weight", (32, 4), BF16, payload)],
                   kvs=[kv_str("general.architecture", "qwen3moe")])
        out = self.root / "token-embd.gguf"
        rc, log = self.run_tool(src, out)
        self.assertEqual(rc, 0, log)
        self.assertIn("32 x 4 BF16 (256 B)", log)
        self.check_output(out, [32, 4], BF16, payload, src.name)

    def test_iq4_xs(self):
        payload = pattern(136)                              # one 256-value IQ4_XS block
        src = self.root / "model-iq4xs.gguf"
        write_gguf(src, [("token_embd.weight", (32, 8), IQ4_XS, payload)])
        out = self.root / "token-embd.gguf"
        rc, log = self.run_tool(src, out)
        self.assertEqual(rc, 0, log)
        self.assertIn("32 x 8 IQ4_XS (136 B)", log)
        self.check_output(out, [32, 8], IQ4_XS, payload, src.name)

    def test_source_shard_copies_only_the_embedding(self):
        payload = pattern(32 * 4 * 2)
        src = self.root / "model-00002-of-00004.gguf"
        # a later shard of a split: no general.architecture, the embedding sits between other tensors
        write_gguf(src, [("blk.0.attn_q.weight", (16, 2), BF16, pattern(64)),
                         ("token_embd.weight", (32, 4), BF16, payload),
                         ("output_norm.weight", (4,), F32, pattern(16))],
                   kvs=[kv_u16("split.no", 1), kv_u16("split.count", 4)], tail=b"after the last tensor")
        before = src.read_bytes()
        out = self.root / "token-embd.gguf"
        rc, log = self.run_tool(src, out)
        self.assertEqual(rc, 0, log)
        self.check_output(out, [32, 4], BF16, payload, src.name)
        self.assertEqual(src.read_bytes(), before)

    def test_missing_tensor_is_refused(self):
        src = self.root / "model.gguf"
        write_gguf(src, [("output_norm.weight", (8,), F32, pattern(32))])
        out = self.root / "token-embd.gguf"
        rc, log = self.run_tool(src, out)
        self.assertEqual(rc, 1, log)
        self.assertIn("has no token_embd.weight", log)
        self.assertFalse(out.exists())

    def test_non_2d_is_refused(self):
        src = self.root / "model.gguf"
        write_gguf(src, [("token_embd.weight", (32,), BF16, pattern(64))])
        out = self.root / "token-embd.gguf"
        rc, log = self.run_tool(src, out)
        self.assertEqual(rc, 1, log)
        self.assertIn("not 2-D", log)
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
