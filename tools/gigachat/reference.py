#!/usr/bin/env python3
"""PyTorch CPU reference for the GigaChat Audio Conformer encoder.

This is a test oracle, not the runtime.  It loads the original HF module
(``modeling_gigachat_audio.py``) from the checkpoint directory and runs the
shipped code paths -- ``AudioFeatureExtractor`` for the log-mel front end,
``ConformerEncoder`` for the 24 layers, ``ModalityAdapter`` for stacking +
projection -- so that the C++ port (``strata-audio``) is compared against the
reference implementation itself, not against a re-implementation of it.

    python tools/gigachat/reference.py --wav clip10s.wav --out clip10s.npz
    python tools/gigachat/reference.py --wav clip10s.wav --out clip10s.npz --layers 0 11 23 --stages 0

Output (npz):
    mel          float32 (T_mel, 64)   log-mel front end output
    pre_encode   float32 (T_enc, 1024) conv subsampling output
    layer_NN     float32 (T_enc, 1024) output of ConformerLayer NN (with --layers)
    stage_NN_*   float32               sub-module outputs of layer NN (with --stages)
    enc_out      float32 (T_enc, 1024) encoder output (layer 23, no out_proj)
    rows         float32 (L, 1536)     adapter output = the rows the decoder reads
    t_mel, t_enc, n_rows               int64 scalars

Everything is float32 on CPU.  The shipped checkpoint is F32 on disk; the model
card's bf16 cast is applied later, at the decoder.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
import types

import numpy as np


def load_module(modeling_path: str) -> types.ModuleType:
    """Import modeling_gigachat_audio.py by path (it is not an installed package)."""
    spec = importlib.util.spec_from_file_location("gigachat_audio_modeling", modeling_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {modeling_path}")
    mod = importlib.util.module_from_spec(spec)
    # Python >= 3.13 dataclasses look the class up in sys.modules to resolve annotations
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_modules(mod, ckpt_dir: str):
    """Build the encoder + adapter exactly as the checkpoint's config describes."""
    import json
    from safetensors.torch import load_file

    cfg = json.load(open(os.path.join(ckpt_dir, "config.json")))
    enc_cfg = dict(cfg["encoder_config"])
    enc_cfg["max_position_seq_len"] = 131072   # ConformerConfig default; not in config.json
    encoder = mod.build_acoustic_encoder(mod.build_encoder_config(enc_cfg))
    adapter = mod.build_modality_adapter(cfg["projector_config"], cfg["subsampler_config"])

    sd = load_file(os.path.join(ckpt_dir, "audio_encoder.safetensors"))
    enc_sd, adp_sd = mod._split_by_prefix(sd)
    missing, unexpected = encoder.load_state_dict(enc_sd, strict=True), None
    del missing, unexpected
    adapter.load_state_dict(adp_sd, strict=True)
    return cfg, encoder.eval(), adapter.eval()


def mel_from_wav(mod, cfg: dict, wav_path: str):
    """The shipped front end: downmix, resample, torchaudio mel, log(clamp).

    Also returns the same waveform in float64, downmixed and resampled the same way, so the
    reference can report a float64 mel beside its float32 one.
    """
    import soundfile as sf
    import torch
    import torchaudio

    data, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    wav = torch.from_numpy(np.asarray(data))
    if wav.ndim > 1:
        wav = wav.mean(dim=1)
    wav = wav.to(torch.float32)
    fe = mod.AudioFeatureExtractor(cfg.get("feature_extractor"))
    mel = fe.process_waveform(wav, int(sr))

    w64 = torch.from_numpy(np.asarray(data, dtype=np.float64))
    if w64.ndim > 1:
        w64 = w64.mean(dim=1)
    if int(sr) != 16000:
        w64 = torchaudio.functional.resample(w64, int(sr), 16000)
    return mel.contiguous(), int(sr), wav, w64.contiguous()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="directory with config.json, modeling_gigachat_audio.py, audio_encoder.safetensors")
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--layers", type=int, nargs="*", default=[],
                    help="ConformerLayer indices whose outputs to save")
    ap.add_argument("--stages", type=int, nargs="*", default=[],
                    help="layer indices whose sub-module outputs to save (ffn1/attn/conv/ffn2/norm)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--dtype", default="float32", choices=["float32"])
    args = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    if args.threads:
        torch.set_num_threads(args.threads)

    t0 = time.time()
    mod = load_module(os.path.join(args.ckpt, "modeling_gigachat_audio.py"))
    cfg, encoder, adapter = build_modules(mod, args.ckpt)
    print(f"reference: weights loaded in {time.time() - t0:.1f} s", file=sys.stderr)

    t0 = time.time()
    mel, sr, wav, wav64 = mel_from_wav(mod, cfg, args.wav)
    print(f"reference: mel {tuple(mel.shape)} from {wav.numel()} samples @ {sr} Hz in "
          f"{time.time() - t0:.2f} s", file=sys.stderr)

    out = {}
    hooks = []
    if args.stages:
        def mk(name):
            def hook(_m, _inp, o):
                out[name] = o.detach().float().cpu().numpy()
            return hook
        for li in args.stages:
            layer = encoder.layers[li]
            hooks.append(layer.feed_forward1.register_forward_hook(mk(f"stage_{li:02d}_ffn1")))
            hooks.append(layer.self_attn.register_forward_hook(mk(f"stage_{li:02d}_attn")))
            hooks.append(layer.conv.register_forward_hook(mk(f"stage_{li:02d}_conv")))
            hooks.append(layer.feed_forward2.register_forward_hook(mk(f"stage_{li:02d}_ffn2")))
            hooks.append(layer.norm_out.register_forward_hook(mk(f"stage_{li:02d}_norm_out")))
    for li in args.layers:
        hooks.append(encoder.layers[li].register_forward_hook(
            lambda _m, _i, o, li=li: out.__setitem__(f"layer_{li:02d}", o[0].detach().float().numpy())))

    # The encoder's own entry points, so a divergence can be attributed to a stage.
    hooks.append(encoder.pre_encode.register_forward_hook(
        lambda _m, _i, o: out.__setitem__("pre_encode", o[0][0].detach().float().numpy())))

    t0 = time.time()
    with torch.no_grad():
        enc_out, lens = encoder(mel.unsqueeze(0), torch.tensor([mel.shape[0]]))
        rows, rlens = adapter(enc_out, lens)
    dt = time.time() - t0
    for h in hooks:
        h.remove()

    # A float64 version of the same front end: torchaudio's shipped mel runs its STFT in float32,
    # which puts ~1e-4 of its own rounding on small mel values.  Comparing against this shows which
    # side of a mel difference is the port and which side is the reference's f32 FFT.
    import torchaudio
    win64 = torch.hann_window(400, periodic=True, dtype=torch.float64)
    stft = torch.stft(wav64, n_fft=400, hop_length=160, win_length=400, window=win64,
                      center=True, pad_mode="reflect", normalized=False, onesided=True,
                      return_complex=True)
    p64 = (stft.real ** 2 + stft.imag ** 2)
    fb64 = torchaudio.functional.melscale_fbanks(201, 0.0, 8000.0, 64, 16000, norm=None,
                                                 mel_scale="htk").double()
    mel64 = torch.log(torch.clamp(p64.transpose(0, 1) @ fb64, 1e-9, 1e9))
    out["mel64"] = mel64.numpy()
    out["mel"] = mel.numpy()
    out["enc_out"] = enc_out[0].float().numpy()
    out["rows"] = rows[0].float().numpy()
    out["t_mel"] = np.int64(mel.shape[0])
    out["t_enc"] = np.int64(enc_out.shape[1])
    out["n_rows"] = np.int64(rows.shape[1])
    print(f"reference: encoder {tuple(enc_out.shape)} + adapter {tuple(rows.shape)} in {dt:.2f} s",
          file=sys.stderr)
    print(f"reference: lengths mel={int(lens[0])} adapter={int(rlens[0])}", file=sys.stderr)
    assert int(lens[0]) == enc_out.shape[1], "encoder returned fewer frames than it computed"
    assert int(rlens[0]) == rows.shape[1], "adapter returned fewer rows than it computed"

    np.savez(args.out, **out)
    print(f"reference: wrote {args.out} ({os.path.getsize(args.out)} B)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
