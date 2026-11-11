#!/usr/bin/env python3
"""tools/nvfp4_synth_pack.py - a synthetic NVFP4 experts.bin for mmq_nvfp4_parity, without a model or a download.

    python tools/nvfp4_synth_pack.py --out build/nvfp4-synth/experts.bin
    python tools/nvfp4_synth_pack.py --out one.bin --experts 1 --seed 7

The NVFP4 routed-expert pack (tools/iq_pack.py) can only be written from an NVFP4 GGUF that carries ModelOpt's
per-expert `.scale` tensors, and `tools/nvfp4_convert.py` needs the ModelOpt checkpoint. The prompt path's numerics do
not: `mmq_nvfp4_parity` (src/prefill/mmq_nvfp4_parity.cpp) reads weights from an `experts.bin` and makes its own
activations, and its FP64 reference decodes the same bytes with ggml's NVFP4 decoder. So synthetic weights are enough
to run the check on a card with no pack and no download.

Layout per expert blob, the one the parity tool and the engine's NVFP4 loader expect (docs/NVFP4.md):

    gate/up  [1280 x 2560] NVFP4 = 1,843,200 B     ([2 * n_ff x n_embd], gate row 2f, up row 2f + 1)
    down     [2560 x  640] NVFP4 =   921,600 B     ([n_embd x n_ff])
    tail     {s_gate, s_up, s_down, 0} float32 = 16 B
    stride   2,764,816 B  (the constant in src/prefill/mmq_nvfp4_parity.cpp and tools/iq_pack.py)

One NVFP4 block is 64 values: four UE4M3 scales (one per 16-value sub-block) then 32 bytes of E2M1 nibbles; within a
sub-block the low nibbles are the first 8 values and the high nibbles the next 8 (ggml's dequantize_row_nvfp4). The
weights are Gaussian, quantized the way ModelOpt does it: amax/6 per 16 values, the scale rounded to UE4M3, each value
to the nearest E2M1 code. The tail is 1.0/1.0/1.0/0, which is what `--real` multiplies the projections by.

This is not a model pack: nothing can load it as one, and `mmq_nvfp4_parity` is the only consumer. `--group` needs at
least 20 blobs (the tool reads 20 layer-0 experts); the default is 20. Twenty blobs take about 1.5 minutes in Python.

The measured numbers are in docs/PLATFORM_GATES.md, section 4 (MIT, PolyStrata).
"""
import argparse

import numpy as np

N, FF = 2560, 640          # n_embd, n_ff of the compiled geometry (flash_next)
STRIDE = 2764816           # one expert blob in experts.bin

# UE4M3: sign | 4-bit exponent (bias 7) | 3-bit mantissa, no inf and no nan. Index = the byte value 0..127.
UE4M3 = np.array([
    (m / 8.0) * 2.0**-6 if e == 0 else (1 + m / 8.0) * 2.0 ** (e - 7)
    for e in range(16) for m in range(8)
])
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def ue4m3_byte(v: float) -> int:
    """The nearest UE4M3 value's byte (0 for v <= 0)."""
    if v <= 0:
        return 0
    return int(np.argmin(np.abs(UE4M3 - v)))


def quant_nvfp4(w: np.ndarray) -> bytes:
    """w: [rows, cols] float32, cols % 64 == 0 -> the block_nvfp4 bytes ggml decodes."""
    rows, cols = w.shape
    out = bytearray()
    for r in range(rows):
        for b in range(cols // 64):
            blk = w[r, b * 64 : (b + 1) * 64].reshape(4, 16)
            d = np.zeros(4, dtype=np.uint8)
            qs = np.zeros(32, dtype=np.uint8)
            for s in range(4):
                amax = float(np.max(np.abs(blk[s])))
                d[s] = ue4m3_byte(amax / 6.0)
                scale = UE4M3[d[s]] if d[s] else 0.0
                if scale == 0.0:
                    continue
                codes = np.abs(blk[s]) / scale
                idx = np.argmin(np.abs(codes[:, None] - E2M1[None, :]), axis=1)
                idx = np.where(blk[s] < 0, idx | 0x8, idx).astype(np.uint8)
                qs[s * 8 : s * 8 + 8] = (idx[:8] & 0x0F) | ((idx[8:] & 0x0F) << 4)
            out += d.tobytes() + qs.tobytes()
    return bytes(out)


def blob(rng) -> bytes:
    """One expert: gate/up, down, the 16-byte scale tail. The weights are 0.02 * N(0, 1), as a residual stream's."""
    gu = rng.standard_normal((2 * FF, N), dtype=np.float32) * 0.02
    dn = rng.standard_normal((N, FF), dtype=np.float32) * 0.02
    tail = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32)
    b = quant_nvfp4(gu) + quant_nvfp4(dn) + tail.tobytes()
    assert len(b) == STRIDE, (len(b), STRIDE)
    return b


def main() -> int:
    ap = argparse.ArgumentParser(description="Write a synthetic NVFP4 experts.bin for mmq_nvfp4_parity.")
    ap.add_argument("--out", default="experts.bin", help="output file (default: experts.bin)")
    ap.add_argument("--experts", type=int, default=20, help="expert blobs to write (default: 20; --group needs 20)")
    ap.add_argument("--seed", type=int, default=20261111, help="numpy default_rng seed (default: 20261111)")
    a = ap.parse_args()
    if a.experts < 1:
        ap.error("--experts must be at least 1")
    rng = np.random.default_rng(a.seed)
    with open(a.out, "wb") as f:
        for e in range(a.experts):
            f.write(blob(rng))
            print("expert", e, flush=True)
    print(f"wrote {a.experts} blob(s), {a.experts * STRIDE} bytes, to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
