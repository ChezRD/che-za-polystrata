#!/usr/bin/env python3
"""tools/embd_extract.py - copy one GGUF's token embedding into the one-tensor GGUF that --embd-gguf reads.

`token_embd.weight` is stored in a type the native embedding may or may not read: Q2_K (REAP-320) is refused,
IQ4_XS (the base Qwen3.8 shard) is accepted.  This tool copies the tensor's bytes unchanged, so the output has the
same type and values as the source; it is the way to take the embedding from another file of the same model when
the model file itself carries an unsupported one.

    python tools/embd_extract.py --gguf <model-or-shard.gguf> --out <token-embd.gguf>

The source may be shard 1 of a split model: only the tensor table and the one payload are read.  The output uses the
same one-tensor form as tools/embd_bf16_pack.py (`general.architecture = strata-embd`).
"""
import argparse
import pathlib
import struct
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gguf_reader as G  # noqa: E402

GGUF_TYPE_STRING = 8
ALIGN = 32


def gguf_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def kv_string(key: str, val: str) -> bytes:
    return gguf_string(key) + struct.pack("<I", GGUF_TYPE_STRING) + gguf_string(val)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gguf", required=True, help="a model or shard GGUF that holds token_embd.weight")
    ap.add_argument("--out", required=True, help="the one-tensor GGUF to write")
    a = ap.parse_args()

    src = pathlib.Path(a.gguf).absolute()
    out = pathlib.Path(a.out)
    g = G.GGUFFile(src)
    t = next((x for x in g.tensors if x.name == "token_embd.weight"), None)
    if t is None:
        print("%s has no token_embd.weight" % src.name)
        return 1
    if len(t.shape) != 2:
        print("token_embd.weight is not 2-D")
        return 1
    n_embd, n_vocab = int(t.shape[0]), int(t.shape[1])
    n = t.expected_bytes()
    if n is None:
        print("token_embd.weight has an unsupported or truncated payload")
        return 1
    print("token_embd.weight: %d x %d %s (%d B)" % (n_embd, n_vocab, t.type_name, n), flush=True)

    kvs = [kv_string("general.architecture", "strata-embd"),
           kv_string("general.name", "token embedding, copied from " + src.name),
           kv_string("strata.embd.source", src.name)]
    head = b"GGUF" + struct.pack("<IQQ", 3, 1, len(kvs)) + b"".join(kvs)
    head += gguf_string("token_embd.weight") + struct.pack("<I", 2) + struct.pack("<QQ", n_embd, n_vocab)
    head += struct.pack("<I", t.type_id) + struct.pack("<Q", 0)
    head += b"\0" * ((-len(head)) % ALIGN)

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    with open(tmp, "wb") as w, open(src, "rb") as f:
        w.write(head)
        f.seek(g.data_start + t.offset)
        left = n
        while left:
            b = f.read(min(64 << 20, left))
            if not b:
                print("short read")
                return 1
            w.write(b)
            left -= len(b)
    if tmp.stat().st_size != len(head) + n:
        print("size check failed")
        return 1
    tmp.replace(out)
    print("wrote %s (header %d B, %.2f GiB)" % (out, len(head), (len(head) + n) / 2**30))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
