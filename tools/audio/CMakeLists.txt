# tools/audio - strata-audio, the GigaChat Audio Conformer encoder for Strata's multimodal path.
#
# Two builds from one source, in two build directories:
#
#   CPU (default, unchanged): ggml-cpu with q8_0 matmuls, no CUDA context
#     cmake -S tools/audio -B build-audio -G Ninja -DCMAKE_BUILD_TYPE=Release
#     cmake --build build-audio --target strata-audio
#     cp build-audio/bin/strata-audio engine-poly/strata-audio
#
#   CUDA (opt-in): the same graph on the GPU, CPU fallback for ops without a kernel
#     cmake -S tools/audio -B build-audio-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release \
#           -DSTRATA_AUDIO_CUDA=ON -DSTRATA_AUDIO_CUDA_ARCH=89
#     cmake --build build-audio-cuda --target strata-audio-cuda
#     cp build-audio-cuda/bin/strata-audio-cuda engine-poly/strata-audio-cuda
#
# The CUDA build stays a separate binary (and a separate build dir): a single build dir cannot hold both, because
# ggml picks its backends at configure time.  The server uses the CUDA one only when the audio config says
# "gpu": true, and the CPU binary stays the fallback.
#
# Portable (AVX2 baseline) is the default, like tools/vision: a binary built here with -march=native
# dies with an illegal instruction on an older PC.  Build for this machine with -DSTRATA_PORTABLE=OFF.
cmake_minimum_required(VERSION 3.21)
project(strata_audio C CXX)

set(LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/llama.cpp" CACHE PATH "llama.cpp checkout")
option(STRATA_PORTABLE "build for other PCs: AVX2 baseline, static MSVC runtime" ON)
option(STRATA_AUDIO_CUDA "build the CUDA encoder variant (target strata-audio-cuda)" OFF)
set(STRATA_AUDIO_CUDA_ARCH "89" CACHE STRING "CUDA architectures for the GPU build (89 = RTX 4060 Ti, sm_89)")
if(STRATA_PORTABLE AND MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
if(STRATA_PORTABLE)
  set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
  set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
  foreach(_f GGML_AVX GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2)
    set(${_f} ON CACHE BOOL "" FORCE)
  endforeach()
  foreach(_f GGML_AVX512 GGML_AVX512_VBMI GGML_AVX512_VNNI GGML_AVX512_BF16 GGML_AVX_VNNI
             GGML_AMX_TILE GGML_AMX_INT8 GGML_AMX_BF16)
    set(${_f} OFF CACHE BOOL "" FORCE)
  endforeach()
  message(STATUS "strata-audio: portable build (AVX2 baseline, runs on other PCs)")
else()
  set(GGML_NATIVE ON CACHE BOOL "" FORCE)
  set(GGML_OPENMP ON CACHE BOOL "" FORCE)
  foreach(_f GGML_AVX GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2)
    set(${_f} OFF CACHE BOOL "" FORCE)
  endforeach()
  message(STATUS "strata-audio: native build for this CPU (STRATA_PORTABLE=OFF) - not for other PCs")
endif()
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)

if(STRATA_AUDIO_CUDA)
  # The GPU variant: ggml-cuda is linked into the same binary; the CPU backend stays in it as the fallback for ops
  # without a CUDA kernel (the scheduler moves just those nodes across, see strata_audio.cpp).
  set(GGML_CUDA ON CACHE BOOL "" FORCE)
  set(GGML_CUDA_FA OFF CACHE BOOL "" FORCE)          # no flash-attention kernels: the graph uses soft_max_ext
  set(GGML_CUDA_NCCL OFF CACHE BOOL "" FORCE)        # one card
  set(GGML_CUDA_GRAPHS OFF CACHE BOOL "" FORCE)      # llama.cpp-only feature, unused by the scheduler path
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES ${STRATA_AUDIO_CUDA_ARCH})
  endif()
  set(_audio_target strata-audio-cuda)
  message(STATUS "strata-audio: CUDA build (arch sm_${STRATA_AUDIO_CUDA_ARCH}), target ${_audio_target}")
else()
  set(GGML_CUDA OFF CACHE BOOL "" FORCE)     # CPU only: the serving GPU must keep its memory
  set(_audio_target strata-audio)
  message(STATUS "strata-audio: CPU build, target ${_audio_target}")
endif()

add_subdirectory(${LLAMA_DIR} llama EXCLUDE_FROM_ALL)

add_executable(${_audio_target} strata_audio.cpp)
target_link_libraries(${_audio_target} PRIVATE ggml)
target_compile_features(${_audio_target} PRIVATE cxx_std_17)
if(STRATA_AUDIO_CUDA)
  target_compile_definitions(${_audio_target} PRIVATE STRATA_AUDIO_CUDA)
endif()
set_target_properties(${_audio_target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
