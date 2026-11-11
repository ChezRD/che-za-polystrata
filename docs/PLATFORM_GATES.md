# Platform gates after the 2026-11-11 sync

The 2026-11-11 engine sync (PolyStrata `0.1.45-poly.9` and the
[Strata](https://github.com/Niko1221/Strata) project's `0.2.13.1`, merge `66a89c6f7`) cannot be checked on
every backend on one box. This page lists the five gates that were left open, what was checked without the hardware,
the exact commands for a machine that has it, and what counts as a pass.

The measurements below were taken on an RTX 4060 Ti 16 GB (sm_89) with CUDA 13.4. The machine has no Mac,
no ROCm or oneAPI toolchain and no Intel GPU. Nothing below was run on hardware it does not have.

| gate | state here | closes on | one-line check |
| --- | --- | --- | --- |
| Metal build and kernels | not built, not run | any Apple Silicon Mac, macOS 15+ | `ctest --test-dir build-metal`, 74 tests |
| HIP (gfx1100 / gfx1151 / gfx906) | not configured | a Radeon with ROCm 7 | `ctest --test-dir build-hip`, 27 `hip_*` / `poly_hip_*` tests among them |
| SYCL (Intel Arc) | **compiles**, nothing runs (no Intel GPU) | an Arc card with oneAPI | `ctest --test-dir build-sycl` |
| NVFP4 W4A8 prompt numerics | **checked** | - | `mmq_nvfp4_parity`, 0.516-0.546% against FP64 |
| `ncu` counters | normal user blocked by the driver setting; root works | `sudo -n`, or a reboot with one module parameter | `sudo -n ncu --set basic` reports occupancy and DRAM |

## 1. Metal

**What is open.** The merged tree carries hand-merged Metal sources (`metal/kernels/moe.metal`, the `strata-poly` port,
the batch-MTP test). None of it has been compiled or run for this sync. The fork has no macOS CI job today: `.github/`
holds only issue templates and a pull-request template. This change adds a proposed workflow, manual-only until it has
one green run (see CI below).

**A Mac with the merged tree.** The engine needs the Command Line Tools, macOS 15 or newer (the engine keeps its
buffers in an `MTLResidencySet`), and Apple's metal-cpp headers. `setup_mac.py` downloads the pinned zip
(`metal-cpp_26.zip`, SHA-256 `4df3c078...a3a4`) into `third_party/metal-cpp`; by hand:

```sh
# the kernel tree only: needs no metal-cpp and no model, and runs the parity checks
cmake -S metal -B build-metal-check -DCMAKE_BUILD_TYPE=Release
cmake --build build-metal-check -j 8
ctest --test-dir build-metal-check --output-on-failure

# the engine and strata-poly, with the merged tree
curl -fsSLo metal-cpp_26.zip https://developer.apple.com/metal/cpp/files/metal-cpp_26.zip
shasum -a 256 metal-cpp_26.zip      # 4df3c078b9aadcb516212e9cb03004cbc5ce9a3e9c068fa3144d021db585a3a4
unzip -q metal-cpp_26.zip -d third_party     # gives third_party/metal-cpp
cmake -S metal -B build-metal -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DMETAL_CPP_DIR="$PWD/third_party/metal-cpp" \
  -DSTRATA_METAL_POLY=ON
cmake --build build-metal -j 8
ctest --test-dir build-metal --output-on-failure          # 74 tests, serialized on the GPU by RESOURCE_LOCK
ctest --test-dir build-metal -L poly --output-on-failure  # 46 of them (the strata-poly port and its fixtures)
ctest --test-dir build-metal --output-on-failure -R poly_tiny_qwen_batch_mtp
```

`third_party/strata-ggml` is the default ggml source for this build; `-DSTRATA_GGML_DIR` is only needed to point the
build at another llama.cpp checkout. The last command is the batch-MTP gate of PolyStrata `0.1.45-poly.9`: two
sequences in one batch, each with its own draft block, greedy answers equal to the single-sequence run.

**Pass:** all 74 tests pass (with `-DSTRATA_METAL_POLY=OFF` there are 21: the kernel tree and the runtime hello).
A kernel suite passing does not establish model generation: run `strata-poly` on a small GGUF once
(`--tokens 1,2,3 --max-new 8 --prompt-batch 32 --no-draft`) and read the answer.

**Limits.** The matrix-unit path (`MetalPerformancePrimitives.matmul2d`, `mm.metal`, `flash.metal`, `dense_mm.metal`)
needs macOS 26. On macOS 15 those libraries fail to compile at run time and `mm_ready()` returns false, so the older
kernels run and the MPP code is never exercised; a pass on macOS 15 says nothing about it. The tests use generated
fixtures, not real weights, and no NVFP4 pack has been run end to end on a Mac: the hand-merged `GU_SET(T_NVFP4)`
region compiles, nothing shows its numbers.

**Checked without a Mac.** `cmake -S metal` on Linux stops at `metal/CMakeLists.txt:18: metal/ builds on macOS only`,
so a Linux box cannot even configure the project. What runs here:

- all 47 `metal/**/*.metal` files preprocess with `clang++ -E` (with empty stand-ins for `metal_stdlib` and the Metal
  4 headers): balanced conditionals and complete `#include`s, nothing about types or MSL semantics;
- `python3 metal/tools/gen_tq_lut.py --check` exits 0: `metal/kernels/quant.metal` holds the table that
  `third_party/ggml/ggml-common-tq.h` produces;
- `python3 metal/tools/q36_host.py src/qwen35/q36_engine.cu <out>` writes `q36_host.cpp`, `q36_launch.hpp` and
  `q36.metal` (36 KB): the generator of the `strata-q36` Metal target runs, the generated MSL does not compile;
- the top-level project never includes `metal/` (only a comment names it), so the NVIDIA build is unaffected by
  Metal changes;
- `python3 metal/tools/geo_pins.py` counts Flash-Next-specific literals per file at two revisions. It reports 24
  specific hits added to the Metal sources since `f6a3b0c` (`window.cpp` 14, `dflash2.inc` 6, `gr2.metal` 3,
  `qsa.metal` 1). That is an inventory for a Mac review, not a failure;
- `metal/tools/msl_text_check.py` compares two kernel dumps written by a Mac build (`STRATA_METAL_MSL_DUMP`); with no
  Mac there is no first dump, so it cannot run here.

**CI.** A GitHub-hosted macOS job is feasible as a **compile gate only**: the runner is Apple Silicon with macOS 26 and
Xcode 26, but its GPU is a paravirtual device without Metal 2.0 support (PyTorch's MPS check refuses it on those
runners), so the kernel tests have no usable device there. The
proposed `.github/workflows/macos.yml` builds the engine with `-DSTRATA_METAL_POLY=ON`, runs the 12 tests that carry no
`gpu` label, and runs `ctest -L gpu` with `continue-on-error` for the log. It has not been run. Kernel validation
stays on a real Mac.

## 2. HIP

**What is open.** No HIP build of the merged tree. `cmake -S . -DSTRATA_ENABLE_HIP=ON` stops at
`cmake/hip_backend.cmake:45` with `Failed to find HIP root directory`; there is no `hipcc`, no `/opt/rocm`, no
`/dev/kfd`. An integrated Radeon (gfx1103) is present but is not a build target here, and nothing
here can compile for it. A compile-only build would be possible with AMD's TheRock wheels (about 10 GB, what
`setup.py` installs into `.venv`); that download was not made.

**gfx1100 (RX 7900 XT / XTX), with a system ROCm 7 or setup's wheels:**

```sh
cmake -S . -B build-hip -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DSTRATA_BUILD_TESTS=ON -DSTRATA_PREFILL_MMQ=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100
cmake --build build-hip -j 8
ctest --test-dir build-hip --output-on-failure --timeout 600
ctest --test-dir build-hip -R 'hip_|poly_hip' --output-on-failure   # the 27 hip_* and poly_hip_* tests
./build-hip/strata-device --list-devices && ./build-hip/strata-device --selftest
```

**gfx1151 (Ryzen AI Max, Strix Halo), ROCm from AMD's TheRock tarball** (see [STRIX_HALO.md](STRIX_HALO.md)):

```sh
SDK=$PWD/install          # therock-dist-linux-gfx1151-7.14.1.tar.gz extracted here
cmake -S . -B build-halo -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_PREFILL_MMQ=ON \
  -DSTRATA_BUILD_TESTS=ON -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_HIP_COMPILER=$SDK/lib/llvm/bin/clang++ -DCMAKE_HIP_COMPILER_ROCM_ROOT=$SDK \
  "-DCMAKE_PREFIX_PATH=$SDK;$SDK/lib/rocm_sysdeps;$SDK/lib/llvm" \
  "-DCMAKE_HIP_FLAGS=--rocm-path=$SDK --rocm-device-lib-path=$SDK/lib/llvm/amdgcn/bitcode"
cmake --build build-halo -j 8
ctest --test-dir build-halo --output-on-failure
export STRATA_HIPBLASLT_TUNING=$PWD/tools/hip/gfx1151-hipblaslt-100401.txt   # ROCm 7.14.1's hipBLASLt 100401
```

**gfx906 (MI50 / MI60 / Radeon VII), wave64 compat build, in the community image:**

```sh
docker run --rm -it --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --ulimit memlock=-1 \
  -v "$PWD:/src" mixa3607/rocm-gfx906:7.14-complete
# inside the image
apt-get update && apt-get install -y cmake ninja-build git python3 build-essential
cmake -S /src -B /src/build-906 -G Ninja -DSTRATA_HIP_GFX906=ON -DCMAKE_HIP_ARCHITECTURES=gfx906 \
  -DSTRATA_BUILD_TESTS=ON \
  -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++
ninja -C /src/build-906 strata
ctest --test-dir /src/build-906 --output-on-failure
```

**Pass:** the engine builds for the card's architecture; `strata-device --selftest` reports the card;
`ctest` passes except the known dependencies (`ple_parity` needs the Q2_0 pack fixture, `expert_multi_test` an AVX-512
CPU, `hip_prefill_hipblaslt_gemm` / `hip_prefill_hcd_exact_parity` skip without a tuning table). On gfx11 also run
`./build-hip/hip_gdn_chunked --bench-default` and `--bench` three times each and compare with the 1.6x / 32K figures in
[AMD_HIP.md](AMD_HIP.md).

**What the sync changed under HIP and is untested.** All of these are compiled as HIP in a HIP build:

- `include/strata/core/geometry.hpp` and `include/strata/core/geometries/*` (new; `flash_next` is the default, and the
  default adds no flag, so the compiled shape is the one the previous tree had). Thirteen `src/kernels/cuda/*.cu` files
  include it;
- the geometry rewrite of `src/prefill/kernels.cu` (route `KU` loop, `blob_dequant_kernel`, SwiGLU, `moe_combine`),
  `src/prefill/gemm.cu`, `src/prefill/moe_fused.cu`, `moe_fused_iq.cu`, `moe_mmq.cu`, `moe_f16_common.cuh`;
- `src/prefill/gdn_chunk.cu` and `gdn_tc.cuh`: the gfx11 chunked GDN behind `STRATA_GDN_CHUNKED=1`, whose scratch-plan
  `static_assert` now depends on the geometry;
- `cmake/backends/hip.cmake` (new, 150 lines) and the gfx906 toolchain/source macros in it: a CMake scope or ordering
  error would only show on an AMD machine.

## 3. SYCL

**What is open.** No Intel GPU here: `lspci` lists the NVIDIA card and the AMD integrated Radeon. The host also
has no SYCL compiler (`icpx` and `dpcpp` are absent; `/opt/intel/oneapi` is the 2026.0 runtime with no front end, and
`sycl-ls` prints `No platforms found`), so the engine is built in the `strata-sycl-dev` Docker image.

**Compile-only check (done here, on the merged tree).** The image build took about 20 minutes here, most of it
the base image pull (15.4 GB on disk, oneAPI 2026.1). `sycl/tools/build.sh strata` then compiles the whole port without
a GPU in about 4 minutes:

```sh
docker build -t strata-sycl-dev sycl/tools
H=$(basename "$PWD")
docker run --rm -u $(id -u):$(id -g) -v "$PWD/..:/work" -e REPO=/work/$H -e BUILD_DIR=/work/$H/build-sycl \
  -e JOBS=10 strata-sycl-dev "cd /work/$H && bash sycl/tools/build.sh strata"
# -> [137/137] Linking CXX executable strata, BUILD EXIT 0, errors: 0
```

`-e AOT=bmg-g31` (or `bmg-g21`) compiles the device code ahead of time instead of SPIR-V JIT; it needs `ocloc` and
`build.sh` refuses an ocloc known to crash on the Battlemage kernels (#3219). The AOT path was not run here.

The test and tool binaries all link (`cmake --build . -j 10 -- -k 0`, 124 targets). `ctest` registers 82 tests. The 11
that look host-only were run: 8 pass (`gated_proposal_test`, the four `spec_admission_*`, `short_cache_test`,
`pipeline_auto_test`, `slot_history_test`), 1 skips (`conversation_snapshot_test`), and 2 turn out to create a queue and
abort on it (`resident_plan_parity`, `verify_refusal_test`). Everything else stops at the queue:

```
No device of requested type available. Please check https://software.intel.com/.../intel-oneapi-dpcpp-system-requirements.html
terminate called after throwing an instance of 'sycl::_V1::exception'
```

The project has no `strata-poly` target: the SYCL port is the Qwen engine, not the smaller families.

**On an Intel Arc machine** (oneAPI 2025.3 or newer with oneMKL), run on the host, where the driver's Level Zero is:

```sh
source /opt/intel/oneapi/setvars.sh
cmake -S sycl -B build-sycl -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DSTRATA_SYCL_AOT=bmg-g31          # Pro B70; bmg-g21 for B580 / B570 / Pro B60; empty for JIT (Arc A-series)
cmake --build build-sycl --target strata -j 12
ctest --test-dir build-sycl --output-on-failure
```

To run the tests inside the container on such a machine, add `--device=/dev/dri --group-add render --ipc=host`.

**Pass:** `ctest` passes on the card, and a short server run answers. [INTEL.md](INTEL.md) and
[INTEL_ARC.md](INTEL_ARC.md) hold the measured numbers.

**Limits.** A clean compile says nothing about correctness or speed, and the container's Python has no numpy, so
`iq_parity` and its fixtures were not registered there. 33 files under `sycl/` changed in the merge (+550/−1407),
including the migrated `sycl/src/prefill/*.dp.cpp` and `sycl/src/kernels/cuda/*.dp.cpp`; the geometry header now
reaches `sycl/src/prefill` too. None of it ran on a card.

## 4. NVFP4 W4A8 prompt numerics (closed here)

The prompt path's `w4a8` products can be checked without a model pack or a download: `mmq_nvfp4_parity` reads weights
from an `experts.bin` and generates its own activations, and the reference decodes the same bytes with ggml.

`tools/iq_pack.py` writes an NVFP4 pack (type 40, a 16-byte `{s_gate, s_up, s_down, 0}` tail after each blob) but only
from an NVFP4 GGUF that carries ModelOpt's `.scale` tensors; it cannot turn another quantization into NVFP4, and
`tools/nvfp4_convert.py` needs the ModelOpt checkpoint. A synthetic `experts.bin` needs neither:
`tools/nvfp4_synth_pack.py` writes one with the same block layout and tail, from Gaussian weights, in about 1.5 minutes.

```sh
mkdir -p build/nvfp4-synth
python3 tools/nvfp4_synth_pack.py --out build/nvfp4-synth/experts.bin
STRATA_PREFILL_NVFP4=w4a8 ./build-sync/mmq_nvfp4_parity build/nvfp4-synth/experts.bin
STRATA_PREFILL_NVFP4=w4a8 ./build-sync/mmq_nvfp4_parity build/nvfp4-synth/experts.bin --group
```

Measured here (RTX 4060 Ti, merged tree): gate/up and down at 1, 7, 37, 128 and 300 rows, 0.516-0.546% relative
Frobenius error against the FP64 reference; the prompt path's launch shape (`--group`, 508 rows over 20 experts)
0.5337%, 0.5359% and 0.5350%. [NVFP4.md](NVFP4.md) states 0.53% for `w4a8`; the numbers agree. `w4a4` on this card
prints `needs an sm_12x card in a 12x build; W4A8` and runs the same products.

The real-prompt gate still needs a pack and a model:

```sh
STRATA_DUMP_MOE_INPUT=1 ./build-sync/strata ... --prefill 4096      # dumps one layer's MoE input
STRATA_PREFILL_NVFP4=w4a8 ./build-sync/mmq_nvfp4_parity <pack>/experts.bin --real moe_input.bin 20
```

**Pass:** `RESULT: ok`, and gate/up near 0.46% and down near 0.90% on real rows, the numbers in [NVFP4.md](NVFP4.md).

## 5. Nsight Compute counters

`ncu` is installed (2026.3.1.0). The normal user is refused:
`ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0.`
`/proc/driver/nvidia/params` has `RmProfilingAdminOnly: 1` (the module parameter is
`NVreg_RestrictProfilingToAdminUsers`). `perf_event_paranoid=2` is unrelated.

Passwordless sudo is available here, so the counters are reachable with `sudo -n`:

```sh
sudo -n ncu --set basic --kernel-name regex:native_down_multi_kernel --launch-count 1 \
  ./build-sync/nvfp4_decode_bench
```

The decode down kernel: 284.45 us, DRAM throughput 86.64%, SM throughput 40.36%, achieved occupancy
61.90% (theoretical 66.67%), 64 registers, 37.65 waves per SM. The first three launches of the same
bench report quantize_q8_1 2.30 us / 25.43% DRAM, native_gu_multi 508.19 us / 94.10%, and swiglu_q8_1
5.25 us / 50.19%. So `ncu` is not blocked here; only the non-root user is.

`nsys` works without counters (2026.3.2). Its summaries replace the timing and copy parts of `ncu --set basic`:

```sh
nsys profile --stats=true -o build/nvfp4-nsys ./build-sync/nvfp4_decode_bench
nsys stats --report cuda_gpu_kern_sum,cuda_gpu_kern_gb_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,cuda_api_sum \
  build/nvfp4-nsys.nsys-rep
```

`cuda_gpu_kern_sum` gives per-kernel time and launch count, `cuda_gpu_kern_gb_sum` the grid and block shape,
`cuda_gpu_mem_time_sum` / `cuda_gpu_mem_size_sum` the copies, `cuda_api_sum` the launch and sync cost. There is no
substitute for SM or DRAM utilization, occupancy and warp stalls: those need the counters. For a non-root user,
add one line to `/etc/modprobe.d/nvidia.conf` and reboot:

```sh
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiler.conf
sudo reboot
grep RmProfilingAdminOnly /proc/driver/nvidia/params    # 0 after the reboot
```

A container with `--gpus all --privileged -e NVIDIA_DRIVER_CAPABILITIES=compute,utility` also counts as admin.
