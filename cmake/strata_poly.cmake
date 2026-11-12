# cmake/strata_poly.cmake - the fork's PolyStrata targets, ported to the 0.2.13 build layout.
#
# The 0.2.13 top level is cut by backend: cmake/backends/*.cmake, cmake/strata_ggml.cmake and
# cmake/strata_gpu_engine.cmake only DEFINE macros, which CMakeLists.txt calls where the block used to be.  This file
# follows the same pattern: it defines strata_poly_targets(), called after strata_cpu_tests() and before
# strata_cuda_spill_guard(), so the 35 fork-only targets (strata-poly, strata_poly_mmq, strata_gdn_mma, local_hooks_test
# and the poly tests) keep the position they had in the fork's single-file CMakeLists.txt (main:CMakeLists.txt:2984-3299).
#
# Differences against that file, and only these:
#   * ${strata_llamacpp_SOURCE_DIR} (the fork's name for the llama.cpp checkout) is ${_strata_ggml_root}, which
#     cmake/strata_ggml.cmake sets to third_party/strata-ggml or to -DSTRATA_GGML_DIR.
#   * ${_strata_gg} is ${_strata_ggml_root}/ggml, and the ${_strata_mmq} fallback is
#     ${_strata_ggml_root}/ggml/src/ggml-cuda.
#   * the 20 MMQ instance wrappers come from third_party/poly-mmq-instances, not from the ggml tree's
#     template-instances: the vendored tree ships 14 instances and lacks q4_0 q4_1 q2_k q3_k iq1_s mxfp4 nvfp4.
#     third_party/poly-mmq-instances/README.md pins llama.cpp 3cf03257 (MIT).  ${_strata_mmq} stays on the include
#     path, because each wrapper includes "mmq.cuh".
#   * every block is if(...) ... endif(), never if(NOT ...) return(): in CMake a return() inside a macro returns from
#     the caller's scope, which would skip the rest of the top-level file.
#
# PolyStrata 0.1.45-poly.9's build additions are carried here as they are in main: src/poly/mmq_nvfp4.cu in the
# MMQ library, STRATA_POLY_NVFP4=1 on CUDA, the NVFP4 MMQ instance in the 12x -> 12xa object library, and the
# batch-MTP Metal test (metal/CMakeLists.txt, not this file).

set(_strata_poly_root "${CMAKE_CURRENT_LIST_DIR}/..")

macro(strata_poly_targets)
  find_package(Threads REQUIRED)

  if(STRATA_BUILD_CONVERSATION_TESTS)
    add_executable(glm_conversations_test ${_strata_poly_root}/src/poly/conversations_test.cpp
                   ${_strata_poly_root}/src/poly/spill.cpp)
    add_test(NAME glm_conversations_test COMMAND glm_conversations_test)
    add_executable(poly_guesses_test ${_strata_poly_root}/src/poly/guesses_test.cpp
                   ${_strata_poly_root}/src/spec/draft_policy.cpp)
    target_include_directories(poly_guesses_test PRIVATE ${_strata_poly_root}/include)
    add_test(NAME poly_guesses_test COMMAND poly_guesses_test)
  endif()

  # The opt-in lifecycle is a host-only test, including on non-CUDA builds.
  if(STRATA_BUILD_TESTS)
    add_executable(poly_hip_value_split_policy_test ${_strata_poly_root}/src/poly/hip_value_split_policy_test.cpp)
    add_test(NAME poly_hip_value_split_policy_test COMMAND poly_hip_value_split_policy_test)
    add_executable(poly_gdn_mma_control_test ${_strata_poly_root}/src/poly/gdn_mma_control_test.cpp)
    add_test(NAME poly_gdn_mma_control_test COMMAND poly_gdn_mma_control_test)
    add_executable(poly_stream_stage_test ${_strata_poly_root}/src/poly/stage_copy_test.cpp)
    add_test(NAME poly_stream_stage_test COMMAND poly_stream_stage_test)
  endif()

  # Fork-only: the hooks in upstream files (IQ1_S type 19, IQ4_NL gate/up) that a merge must keep.
  # Guarded as in main:CMakeLists.txt:472/1505, by the GPU backend alone: the fork builds it with the tests off too.
  if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)
    add_executable(local_hooks_test ${_strata_poly_root}/src/kernels/local_hooks_test.cpp)
    target_include_directories(local_hooks_test PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(local_hooks_test PRIVATE strata_kernels ${_strata_gpu_runtime_target})
    add_test(NAME local_hooks_test COMMAND local_hooks_test)
  endif()

  # Shared prepared-input GDN recurrence. Either engine may link this library;
  # only the Poly Qwen35 adapter is integrated initially. Keep FP32/BF16 hi+lo math
  # separate from the MMQ targets' -use_fast_math compilation options.
  if(STRATA_NATIVE_EXPERTS AND STRATA_ENABLE_CUDA)
    set(_strata_gdn_ggml ${_strata_ggml_root}/ggml)
    add_library(strata_gdn_mma STATIC ${_strata_poly_root}/src/kernels/cuda/gdn_mma.cu)
    target_include_directories(strata_gdn_mma PUBLIC ${_strata_poly_root}/include
      PRIVATE ${_strata_gdn_ggml}/include ${_strata_gdn_ggml}/src ${_strata_gdn_ggml}/src/ggml-cuda)
    target_link_libraries(strata_gdn_mma PRIVATE ggml-base ${_strata_gpu_runtime_target})
    if(STRATA_BUILD_TESTS)
      add_executable(poly_gdn_mma_policy_test ${_strata_poly_root}/src/kernels/gdn_mma_policy_test.cpp)
      target_include_directories(poly_gdn_mma_policy_test PRIVATE ${_strata_poly_root}/include)
      add_test(NAME poly_gdn_mma_policy_test COMMAND poly_gdn_mma_policy_test)
      add_executable(gdn_mma_test ${_strata_poly_root}/src/kernels/gdn_mma_test.cu)
      target_link_libraries(gdn_mma_test PRIVATE strata_gdn_mma ${_strata_gpu_runtime_target})
      add_test(NAME gdn_mma_test COMMAND gdn_mma_test)
      set_tests_properties(gdn_mma_test PROPERTIES SKIP_RETURN_CODE 77)
    endif()
  endif()

  # ---- strata-poly: GLM 5.3 Flash (llama.cpp's glm5-next), read from its GGUF ----
  #
  # A second engine beside `strata`.  It shares the quantized matvec kernels and the GGUF reader, and takes the
  # CPU experts' dot products from ggml-cpu.  On HIP it is the same CUDA-shaped code through the hip_compat headers
  # (the wave32 RDNA backend; gfx906's wave64 compat layer is not supported here).
  if(STRATA_NATIVE_EXPERTS AND (STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP) AND NOT STRATA_HIP_GFX906)
    # The prompt path's quantized products: llama.cpp's MMQ kernels (ggml-cuda, MIT) for the types the model's
    # GGUFs keep their matrices in (the list in src/poly/mmq.cu), from the checkout strata_mmq is compiled from and
    # with the same host symbols.  A library of its own, so that what `strata` multiplies with MMQ stays as it is.
    if(NOT _strata_mmq)   # a HIP build without STRATA_PREFILL_MMQ has not set these up yet
      set(_strata_gg ${_strata_ggml_root}/ggml)
      set(_strata_mmq ${_strata_gg}/src/ggml-cuda)
    endif()
    set(_strata_poly_mmq_instances "")
    foreach(_t q4_0 q4_1 q5_0 q5_1 q8_0 q2_k q3_k q4_k q5_k q6_k iq1_s iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl
               iq4_xs)
      list(APPEND _strata_poly_mmq_instances
           ${_strata_poly_root}/third_party/poly-mmq-instances/mmq-instance-${_t}.cu)
    endforeach()
    # MXFP4 and NVFP4: on RTX 50 llama.cpp's tiles use the FP4 MMA with block scales, which exists only in the
    # arch-specific target, so on CUDA their instances alone are built with 12x -> 12xa, as Strata's W4A4 unit is. A
    # Blackwell other than 12.0 has no image of them, and Mmq::fits() leaves MXFP4 out there.  NVFP4 takes this
    # instance only on a 12.0 card with STRATA_PREFILL_NVFP4=w4a4; otherwise its tiles are mmq_nvfp4.cu's int8 ones,
    # built for the architectures as given.  On HIP there is no NVFP4 prompt product.
    set(_strata_poly_mxfp4 ${_strata_poly_root}/third_party/poly-mmq-instances/mmq-instance-mxfp4.cu)
    if(STRATA_ENABLE_HIP)
      list(APPEND _strata_poly_mmq_instances ${_strata_poly_mxfp4})
    else()
      list(APPEND _strata_poly_mxfp4 ${_strata_poly_root}/third_party/poly-mmq-instances/mmq-instance-nvfp4.cu)
      list(APPEND _strata_poly_mmq_instances ${_strata_poly_root}/src/poly/mmq_nvfp4.cu)
      set(_strata_poly_fp4_archs ${CMAKE_CUDA_ARCHITECTURES})
      list(TRANSFORM _strata_poly_fp4_archs REPLACE "^(12[0-9])$" "\\1a")
      list(TRANSFORM _strata_poly_fp4_archs REPLACE "^(12[0-9])-(real|virtual)$" "\\1a-\\2")
      add_library(strata_poly_mmq_mxfp4 OBJECT ${_strata_poly_mxfp4})
      set_target_properties(strata_poly_mmq_mxfp4 PROPERTIES CUDA_ARCHITECTURES "${_strata_poly_fp4_archs}")
      target_include_directories(strata_poly_mmq_mxfp4 PRIVATE ${_strata_gg}/include ${_strata_gg}/src ${_strata_mmq})
      target_compile_options(strata_poly_mmq_mxfp4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -use_fast_math>)
      if(MSVC)
        target_compile_options(strata_poly_mmq_mxfp4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>)
      endif()
      list(APPEND _strata_poly_mmq_instances $<TARGET_OBJECTS:strata_poly_mmq_mxfp4>)
    endif()
    if(STRATA_ENABLE_HIP)
      # CMake does not infer HIP from the .cu suffixes outside src/ (and drops a .cu source it has no language
      # for), and ggml's MMQ sources take its native HIP path (vendors/hip.h) instead of the hip_compat headers,
      # as strata_mmq's does.
      set_source_files_properties(${_strata_mmq}/quantize.cu ${_strata_poly_mmq_instances} PROPERTIES LANGUAGE HIP)
    endif()
    add_library(strata_poly_mmq STATIC ${_strata_poly_root}/src/poly/mmq.cu ${_strata_poly_root}/src/poly/fattn.cu
                ${_strata_poly_root}/src/prefill/ggml_cuda_host.cu ${_strata_mmq}/quantize.cu
                ${_strata_poly_mmq_instances})
    target_include_directories(strata_poly_mmq PRIVATE ${_strata_gg}/include ${_strata_gg}/src ${_strata_mmq}
                                                      ${_strata_poly_root}/third_party/poly-ggml-headers
                                                      ${_strata_poly_root}/include)
    target_compile_options(strata_poly_mmq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -use_fast_math>)
    if(MSVC)
      target_compile_options(strata_poly_mmq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>)
    endif()
    if(STRATA_ENABLE_HIP)
      target_compile_definitions(strata_poly_mmq PRIVATE GGML_USE_HIP=1)
      target_link_libraries(strata_poly_mmq PUBLIC ggml-base hip::host)
    else()
      target_compile_definitions(strata_poly_mmq PRIVATE STRATA_POLY_NVFP4=1)
      target_link_libraries(strata_poly_mmq PUBLIC ggml-base ${_strata_gpu_runtime_target})
    endif()

    if(STRATA_ENABLE_CUDA)
      target_compile_definitions(strata_poly_mmq PUBLIC STRATA_POLY_GDN_MMA=1)
      target_link_libraries(strata_poly_mmq PUBLIC strata_gdn_mma)

      add_executable(poly_gdn_mma_test ${_strata_poly_root}/src/poly/gdn_mma_test.cu ${_strata_poly_root}/src/poly/kernels.cu)
      target_include_directories(poly_gdn_mma_test PRIVATE ${_strata_poly_root}/include)
      target_link_libraries(poly_gdn_mma_test PRIVATE strata_poly_mmq ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
      add_test(NAME poly_gdn_mma_test COMMAND poly_gdn_mma_test)
      set_tests_properties(poly_gdn_mma_test PROPERTIES SKIP_RETURN_CODE 77)
      # Optional model/state check; run by hand with an MTP GGUF and frozen prompt IDs.
      add_executable(poly_gdn_mma_state_test ${_strata_poly_root}/src/poly/gdn_mma_state_test.cpp ${_strata_poly_root}/src/poly/engine.cpp ${_strata_poly_root}/src/poly/glm5next.cpp
                     ${_strata_poly_root}/src/poly/qwen35.cpp ${_strata_poly_root}/src/poly/gemma4.cpp ${_strata_poly_root}/src/poly/deepseek4.cpp ${_strata_poly_root}/src/poly/lfm2moe.cpp
                     ${_strata_poly_root}/src/poly/flint.cpp ${_strata_poly_root}/src/poly/kolibri1.cpp ${_strata_poly_root}/src/poly/gigachat.cpp ${_strata_poly_root}/src/poly/cpu_experts.cpp ${_strata_poly_root}/src/poly/kernels.cu
                     ${_strata_poly_root}/src/poly/kernels_dsv4.cu ${_strata_poly_root}/src/poly/kernels_mla.cu ${_strata_poly_root}/src/core/conversation_memory.cpp)
      target_include_directories(poly_gdn_mma_state_test PRIVATE ${_strata_poly_root}/include ${_strata_gpu_include_directories})
      target_link_libraries(poly_gdn_mma_state_test PRIVATE strata_poly_mmq strata_kernels strata_spec ggml-cpu ggml-base
                           ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target} CUDA::cuda_driver Threads::Threads)
    endif()

    add_executable(strata-poly ${_strata_poly_root}/src/poly/main.cpp ${_strata_poly_root}/src/poly/engine.cpp ${_strata_poly_root}/src/poly/glm5next.cpp ${_strata_poly_root}/src/poly/qwen35.cpp
                              ${_strata_poly_root}/src/poly/gemma4.cpp ${_strata_poly_root}/src/poly/deepseek4.cpp ${_strata_poly_root}/src/poly/lfm2moe.cpp ${_strata_poly_root}/src/poly/flint.cpp
                              ${_strata_poly_root}/src/poly/kolibri1.cpp ${_strata_poly_root}/src/poly/gigachat.cpp ${_strata_poly_root}/src/poly/cpu_experts.cpp ${_strata_poly_root}/src/poly/kernels.cu ${_strata_poly_root}/src/poly/kernels_dsv4.cu
                              ${_strata_poly_root}/src/poly/kernels_mla.cu ${_strata_poly_root}/src/poly/spill.cpp ${_strata_poly_root}/src/core/conversation_memory.cpp)
    target_include_directories(strata-poly PRIVATE ${_strata_poly_root}/include ${_strata_gpu_include_directories})
    # the driver's virtual memory calls: the attention caches grow with the conversation; on HIP they are HIP's
    # memory management calls, which the HIP runtime itself provides (hip_compat/cuda.h)
    if(STRATA_ENABLE_HIP)
      set(_strata_poly_driver "")
    else()
      set(_strata_poly_driver CUDA::cuda_driver)
    endif()
    target_link_libraries(strata-poly PRIVATE strata_poly_mmq strata_kernels strata_spec ggml-cpu ggml-base
                                             ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target}
                                             ${_strata_poly_driver}
                                             Threads::Threads)
    # needs the model file, so it is run by hand: poly_expert_parity MODEL.gguf
    add_executable(poly_expert_parity ${_strata_poly_root}/src/poly/expert_parity.cpp ${_strata_poly_root}/src/poly/cpu_experts.cpp ${_strata_poly_root}/src/poly/kernels.cu)
    target_include_directories(poly_expert_parity PRIVATE ${_strata_poly_root}/include
                                                         ${_strata_gpu_include_directories})
    target_link_libraries(poly_expert_parity PRIVATE strata_poly_mmq strata_kernels ggml-cpu ggml-base
                                                    ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target}
                                                    Threads::Threads)
    # a read with the draft block repeats bit for bit; needs the model file, so it is run by hand:
    # poly_draft_race_test MODEL.gguf [RUNS] [LAYER_SPLIT]
    add_executable(poly_draft_race_test ${_strata_poly_root}/src/poly/draft_race_test.cpp ${_strata_poly_root}/src/poly/engine.cpp ${_strata_poly_root}/src/poly/glm5next.cpp
                                        ${_strata_poly_root}/src/poly/qwen35.cpp ${_strata_poly_root}/src/poly/gemma4.cpp ${_strata_poly_root}/src/poly/deepseek4.cpp
                                        ${_strata_poly_root}/src/poly/lfm2moe.cpp ${_strata_poly_root}/src/poly/flint.cpp ${_strata_poly_root}/src/poly/kolibri1.cpp ${_strata_poly_root}/src/poly/gigachat.cpp
                                        ${_strata_poly_root}/src/poly/cpu_experts.cpp ${_strata_poly_root}/src/poly/kernels.cu ${_strata_poly_root}/src/poly/kernels_dsv4.cu
                                        ${_strata_poly_root}/src/poly/kernels_mla.cu ${_strata_poly_root}/src/core/conversation_memory.cpp)
    target_include_directories(poly_draft_race_test PRIVATE ${_strata_poly_root}/include
                                                           ${_strata_gpu_include_directories})
    target_link_libraries(poly_draft_race_test PRIVATE strata_poly_mmq strata_kernels strata_spec ggml-cpu ggml-base
                                                      ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target}
                                                      ${_strata_poly_driver} Threads::Threads)
    # a prompt batch's routing on the card against the hosts' routing; needs a card, so it is run by hand
    add_executable(poly_route_test ${_strata_poly_root}/src/poly/route_test.cu ${_strata_poly_root}/src/poly/kernels.cu)
    target_include_directories(poly_route_test PRIVATE ${_strata_poly_root}/include
                                                      ${_strata_gpu_include_directories})
    target_link_libraries(poly_route_test PRIVATE ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
    # the fused residual sum, RMS norm and Q8_1 quantization against the three launches; needs a card, run by hand
    add_executable(poly_norm_q8_test ${_strata_poly_root}/src/poly/norm_q8_test.cu ${_strata_poly_root}/src/poly/kernels.cu)
    target_include_directories(poly_norm_q8_test PRIVATE ${_strata_poly_root}/include
                                                        ${_strata_gpu_include_directories})
    target_link_libraries(poly_norm_q8_test PRIVATE strata_kernels ${_strata_gpu_blas_target}
                                                    ${_strata_gpu_runtime_target})
    # a prompt batch's products with few outputs against the same sums on the host; needs a card, run by hand
    add_executable(poly_gemm_small_test ${_strata_poly_root}/src/poly/gemm_small_test.cu ${_strata_poly_root}/src/poly/kernels.cu)
    target_include_directories(poly_gemm_small_test PRIVATE ${_strata_poly_root}/include
                                                           ${_strata_gpu_include_directories})
    target_link_libraries(poly_gemm_small_test PRIVATE ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
    # attend()'s blocks of rows against its other kernels and itself; needs a card, so it is run by hand
    add_executable(poly_attend_test ${_strata_poly_root}/src/poly/attend_test.cu ${_strata_poly_root}/src/poly/kernels.cu)
    target_include_directories(poly_attend_test PRIVATE ${_strata_poly_root}/include
                                                       ${_strata_gpu_include_directories})
    target_link_libraries(poly_attend_test PRIVATE ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
    add_executable(poly_hip_value_split_test ${_strata_poly_root}/src/poly/hip_value_split_test.cu ${_strata_poly_root}/src/poly/kernels.cu)
    target_include_directories(poly_hip_value_split_test PRIVATE ${_strata_poly_root}/include
                              ${_strata_gpu_include_directories})
    target_link_libraries(poly_hip_value_split_test PRIVATE ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
    add_test(NAME poly_hip_value_split_test COMMAND poly_hip_value_split_test)
    set_tests_properties(poly_hip_value_split_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 240)
    # STRATA_HIP_SPLIT_ATTEND (heads of 256 over a q8_0 cache): against an FP64 reference and the old kernels, the position on the card, the
    # fallback and a graph replay; needs a card with 32-lane waves (77 = skipped)
    add_executable(poly_hip_split_attend_test ${_strata_poly_root}/src/poly/hip_split_attend_test.cu ${_strata_poly_root}/src/poly/kernels.cu)
    target_include_directories(poly_hip_split_attend_test PRIVATE ${_strata_poly_root}/include
                              ${_strata_gpu_include_directories})
    target_link_libraries(poly_hip_split_attend_test PRIVATE ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
    add_test(NAME poly_hip_split_attend_test COMMAND poly_hip_split_attend_test)
    set_tests_properties(poly_hip_split_attend_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
    # attend_batch() against attend() with Flint's heads, sinks and window ring; needs a card
    add_executable(poly_sink_attend_test ${_strata_poly_root}/src/poly/sink_attend_test.cu ${_strata_poly_root}/src/poly/kernels.cu)
    target_include_directories(poly_sink_attend_test PRIVATE ${_strata_poly_root}/include
                                                            ${_strata_gpu_include_directories})
    target_link_libraries(poly_sink_attend_test PRIVATE ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
    # STRATA_HIP_PREFILL_FP32 (attend_batch() on FP32 inputs over a q8_0 cache): against an FP64 reference and the default path, and the
    # fallbacks; needs a card (77 = skipped)
    add_executable(poly_hip_prefill_fp32_test ${_strata_poly_root}/src/poly/hip_prefill_fp32_test.cu ${_strata_poly_root}/src/poly/kernels.cu)
    target_include_directories(poly_hip_prefill_fp32_test PRIVATE ${_strata_poly_root}/include
                              ${_strata_gpu_include_directories})
    target_link_libraries(poly_hip_prefill_fp32_test PRIVATE ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
    add_test(NAME poly_hip_prefill_fp32_test COMMAND poly_hip_prefill_fp32_test)
    set_tests_properties(poly_hip_prefill_fp32_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
    # fattn_q8() and fattn_f16() (llama.cpp's flash-attention kernel) against attend_batch(), and with `dec` fdec_q8()
    # against attend(); needs a card, run by hand
    add_executable(poly_fattn_test ${_strata_poly_root}/src/poly/fattn_test.cu ${_strata_poly_root}/src/poly/kernels.cu)
    target_include_directories(poly_fattn_test PRIVATE ${_strata_poly_root}/include ${_strata_poly_root}/src/poly
                                                      ${_strata_gpu_include_directories})
    target_link_libraries(poly_fattn_test PRIVATE strata_poly_mmq ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
    # STRATA_POLY_STAGED_TILES: the staged experts in size classes, each a launch of its own tile width, against one
    # launch, bit for bit; needs a card
    add_executable(poly_mmq_tiles_test ${_strata_poly_root}/src/poly/mmq_tiles_test.cu)
    target_include_directories(poly_mmq_tiles_test PRIVATE ${_strata_poly_root}/include ${_strata_poly_root}/src/poly
                                                          ${_strata_gpu_include_directories})
    target_link_libraries(poly_mmq_tiles_test PRIVATE strata_poly_mmq ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
    add_test(NAME poly_mmq_tiles_test COMMAND poly_mmq_tiles_test)
    set_tests_properties(poly_mmq_tiles_test PROPERTIES SKIP_RETURN_CODE 77)
    # LFM2's short convolution against the same arithmetic on the host; needs a card, so it is run by hand
    add_executable(poly_shortconv_test ${_strata_poly_root}/src/poly/shortconv_test.cu ${_strata_poly_root}/src/poly/kernels.cu)
    target_include_directories(poly_shortconv_test PRIVATE ${_strata_poly_root}/include
                                                          ${_strata_gpu_include_directories})
    target_link_libraries(poly_shortconv_test PRIVATE ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
    # DeepSeek V4's kernels against the same arithmetic on the host; needs a card, so it is run by hand
    add_executable(poly_dsv4_test ${_strata_poly_root}/src/poly/dsv4_test.cu ${_strata_poly_root}/src/poly/kernels_dsv4.cu)
    target_include_directories(poly_dsv4_test PRIVATE ${_strata_poly_root}/include
                                                     ${_strata_gpu_include_directories})
    target_link_libraries(poly_dsv4_test PRIVATE ${_strata_gpu_runtime_target})
    # MLA's absorbed attention against the same arithmetic on the host; synthetic, no model, needs a card
    add_executable(poly_mla_test ${_strata_poly_root}/src/poly/mla_test.cu ${_strata_poly_root}/src/poly/kernels_mla.cu)
    target_include_directories(poly_mla_test PRIVATE ${_strata_poly_root}/include
                                                    ${_strata_gpu_include_directories})
    target_link_libraries(poly_mla_test PRIVATE ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
    # absorbed MLA decode alone, synthetic caches, every context in seconds (no model, no prefill); run by hand
    add_executable(poly_mla_decode_bench ${_strata_poly_root}/bench/micro/mla_decode_bench.cu ${_strata_poly_root}/src/poly/kernels_mla.cu)
    target_include_directories(poly_mla_decode_bench PRIVATE ${_strata_poly_root}/include
                                                            ${_strata_gpu_include_directories})
    target_link_libraries(poly_mla_decode_bench PRIVATE ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
    # the pool of the CPU experts, without a model
    add_executable(poly_workers_test ${_strata_poly_root}/src/poly/workers_test.cpp ${_strata_poly_root}/src/poly/cpu_experts.cpp)
    target_include_directories(poly_workers_test PRIVATE ${_strata_gg}/include)
    target_link_libraries(poly_workers_test PRIVATE ggml-cpu ggml-base Threads::Threads)
    add_test(NAME poly_workers_test COMMAND poly_workers_test)
    # the card's MXFP4 product against ggml's dequantizer on random blocks, without a model; needs a card, run by hand
    add_executable(poly_mxfp4_test ${_strata_poly_root}/src/poly/mxfp4_test.cu)
    target_include_directories(poly_mxfp4_test PRIVATE ${_strata_poly_root}/include ${_strata_gg}/include
                                                      ${_strata_gpu_include_directories})
    target_link_libraries(poly_mxfp4_test PRIVATE strata_kernels ggml-base ${_strata_gpu_runtime_target})
    # the Q8_0 KV cache kernels (store_q8 / store_q8_at) against a host reference (GPU, synthetic, no model)
    if(STRATA_BUILD_TESTS)
      add_executable(poly_kv_q8_test ${_strata_poly_root}/src/poly/kv_q8_test.cpp ${_strata_poly_root}/src/poly/kernels.cu)
      target_include_directories(poly_kv_q8_test PRIVATE ${_strata_poly_root}/include
                                                         ${_strata_gpu_include_directories})
      target_link_libraries(poly_kv_q8_test PRIVATE ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target}
                                                    Threads::Threads)
      add_test(NAME poly_kv_q8_test COMMAND poly_kv_q8_test)
      # the Q4_0 KV cache kernels (store_q4 / store_q4_at) and the Hadamard rotation (fwht_rows) against a host
      # reference (GPU, synthetic, no model)
      add_executable(poly_kv_q4_test ${_strata_poly_root}/src/poly/kv_q4_test.cpp ${_strata_poly_root}/src/poly/kernels.cu)
      target_include_directories(poly_kv_q4_test PRIVATE ${_strata_poly_root}/include
                                                         ${_strata_gpu_include_directories})
      target_link_libraries(poly_kv_q4_test PRIVATE ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target}
                                                    Threads::Threads)
      add_test(NAME poly_kv_q4_test COMMAND poly_kv_q4_test)
    endif()
    if(STRATA_BUILD_TESTS)
      # Phase 0 row 2 (OPTIMIZATION-STACK.md §0.3): every poly KV read path for a geometry, as ns a cell and GB/s,
      # so a model family the engine serves can be measured without editing the bench (a benchmark, no test)
      add_executable(poly_kv_read_bench ${_strata_poly_root}/src/poly/kv_read_bench.cpp
                                         ${_strata_poly_root}/src/poly/kernels.cu)
      target_include_directories(poly_kv_read_bench PRIVATE ${_strata_poly_root}/include ${_strata_poly_root}/src/poly
                                                           ${_strata_gpu_include_directories})
      # strata_poly_mmq already holds fattn.cu (the decode and flash-attention kernels), so it is linked, not rebuilt
      target_link_libraries(poly_kv_read_bench PRIVATE strata_poly_mmq ${_strata_gpu_blas_target}
                                                      ${_strata_gpu_runtime_target} Threads::Threads)
    endif()
    # --kv int8/int8k's kernels (store_int8 / attend_int8) against ggml's own quantizer and against attend() over the
    # same numbers in fp16.  The reference is ggml-cpu's quantize_row_q8_0_ref, hence the ggml links; under HIP the
    # mma tier's probes are behind STRATA_USE_HIP (the dp4a tier is what runs there anyway).
    if(STRATA_BUILD_TESTS)
      add_executable(poly_kv_int8_test ${_strata_poly_root}/src/poly/kv_int8_test.cpp ${_strata_poly_root}/src/poly/kernels.cu)
      target_include_directories(poly_kv_int8_test PRIVATE ${_strata_poly_root}/include
                                                        ${_strata_gpu_include_directories} ${_strata_gg}/include)
      target_link_libraries(poly_kv_int8_test PRIVATE ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target}
                                                    ggml-cpu ggml-base Threads::Threads)
      add_test(NAME poly_kv_int8_test COMMAND poly_kv_int8_test)
    endif()
  endif()
endmacro()
