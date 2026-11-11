# Poly MMQ instances

Twenty `DECL_MMQ_CASE` wrappers that the fork's `strata_poly_mmq` target compiles, one per quant type. The vendored
`third_party/strata-ggml` tree ships only 14 MMQ instance files, and its list is not the fork's list. Keeping the
wrappers here lets the build stop reading instance files from the ggml tree: the ggml tree keeps the implementation,
this directory names the 20 types Poly builds.

## Types

In build order:

```
q4_0 q4_1 q5_0 q5_1 q8_0 q2_k q3_k q4_k q5_k q6_k iq1_s iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs mxfp4 nvfp4
```

Each file is `mmq-instance-<type>.cu` and holds one case:

```c
DECL_MMQ_CASE(GGML_TYPE_<TYPE>);
```

## Origin

Copied from `llama.cpp/ggml/src/ggml-cuda/template-instances` at commit
`3cf03257f219afbe7334045ff7c6a06ac68c627d` (short `3cf03257`). llama.cpp is MIT licensed; see its `LICENSE`.

Only the wrapper is copied. `mmq.cuh` and the MMQ implementation stay in the pinned ggml tree. The build adds that
tree's `ggml-cuda` directory to the include path, so every file here includes `"mmq.cuh"` and not a relative path
into the ggml tree.
