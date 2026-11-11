# poly-ggml-headers

Four llama.cpp `ggml-cuda` headers that `src/poly/fattn.cu` needs and the vendored
`third_party/strata-ggml` tree does not carry. The vendored tree is a partial ggml
(ggml-base, ggml-cpu and the MMQ files); it has `common.cuh`, `mma.cuh`,
`vecdotq.cuh` and `vendors/`, but not the flash-attention headers below.

| file | lines | includes resolved from |
| --- | ---: | --- |
| `cp-async.cuh` | 57 | `common.cuh` (vendored) |
| `convert.cuh` | 66 | `common.cuh` (vendored) |
| `fattn-common.cuh` | 1302 | `common.cuh`, `convert.cuh`, `vecdotq.cuh` (vendored) |
| `fattn-swizzle.cuh` | 126 | `common.cuh`, `mma.cuh` (vendored) |

Origin: `llama.cpp/ggml/src/ggml-cuda/`, pinned at commit
`3cf03257f219afbe7334045ff7c6a06ac68c627d` — the same pin as `third_party/strata-ggml`
and the checkout `setup.py` fetches. MIT; each file keeps its upstream header.

Do not edit these copies. They exist so a plain `cmake` build with the vendored
ggml tree can compile `strata_poly_mmq`; a `-DSTRATA_GGML_DIR=<checkout>` build
uses the checkout's own headers and does not need this directory. `cmake/strata_poly.cmake`
adds it to the `strata_poly_mmq` include path after the ggml-cuda directory, so
the vendored tree wins for every shared header.
