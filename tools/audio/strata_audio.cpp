// tools/audio/strata_audio.cpp - the audio half of Strata's multimodal path.
//
// Turns a WAV into the embeddings the text model reads in place of its [audio_token] ids: the
// GigaChat Audio Conformer encoder (24 layers, d=1024, 200-frame chunk attention with
// pre-projection RoPE base 5000) plus the 4x stacking + 4096->1536 linear adapter, exactly as
// ai-sage/GigaChat3.1-Audio-10B-A1.8B @ bf73d03 computes them.  See
// docs/research/gigachat-audio/03-audio-pipeline.md for the graph this mirrors.
//
//   strata-audio --model <encoder.gguf> [--threads N] [--group-blocks N] [--dump-dir DIR]
//                [--gpu] [--device N]
//
// Resident: prints "READY <n_embd>", then per stdin line
//   ENC <wav> <output>   ->  "OK <n_rows> <t0_ms> <t1_ms> <ms>"  or  "ERR <message>"
//   QUIT
// The output file is one record  int32 {0x31455653 'SVE1', n_rows, nx, ny, n_embd}  (nx = n_rows,
// ny = 1) followed by float32 [n_rows][1536], row i at audio time i * 160 ms.
//
// Two builds: the CPU one (ggml-cpu only, no CUDA context, hides the card) and the CUDA one
// (target strata-audio-cuda, `-DSTRATA_AUDIO_CUDA=ON`).  The CUDA build keeps its CPU backend as the
// fallback: a ggml_backend_sched runs every op with a CUDA kernel on the GPU and moves the few that
// have none across, so the whole 24-layer graph is one scheduler pass.  `--gpu` (or `--device N`)
// picks the card; without it the CUDA binary also runs on the CPU, with the card hidden, exactly as
// the CPU build does.
//
// Streaming is still a later step (this encodes a whole file at once; long files are processed in
// 200-frame-aligned groups with a one-block halo, see encode()).
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"
#ifdef STRATA_AUDIO_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#ifdef STRATA_AUDIO_CUDA
#include <dlfcn.h>
#ifndef _WIN32
#include <unistd.h>
#else
#include <process.h>
#endif
#endif

namespace {

constexpr int64_t kDModel = 1024;      // gca.d_model
constexpr int64_t kHeads = 16;         // gca.n_heads
constexpr int64_t kHeadDim = 64;       // kDModel / kHeads
constexpr int64_t kDff = 4096;         // gca.d_ff
constexpr int64_t kChunk = 200;        // gca.attn_chunk_size (hard block-diagonal attention)
constexpr int64_t kStack = 4;          // gca.stack_factor
constexpr int64_t kFeatIn = 64;        // gca.feat_in
constexpr int64_t kConvK = 5;          // gca.conv_kernel
constexpr int64_t kNMel = 64;
constexpr int64_t kNfft = 400;
constexpr int64_t kHop = 160;
constexpr int64_t kSampleRate = 16000;
constexpr float kNormEps = 1e-5f;      // the encoder's LayerNorm, not the decoder's 1e-6
constexpr float kRopeBase = 5000.0f;   // gca.pos_emb_max_len
constexpr int32_t kSve1Magic = 0x31455653;

void quiet_log(ggml_log_level level, const char* text, void*) {
    if (level >= GGML_LOG_LEVEL_WARN) std::fputs(text, stderr);
}

// ---------------------------------------------------------------- WAV input

struct Wav {
    std::vector<float> samples;   // mono, at sample_rate
    int sample_rate = 0;
    int channels = 0;
    int64_t frames = 0;
};

uint32_t rd_u32(const unsigned char* p) {
    return (uint32_t) p[0] | (uint32_t) p[1] << 8 | (uint32_t) p[2] << 16 | (uint32_t) p[3] << 24;
}
uint16_t rd_u16(const unsigned char* p) { return (uint16_t) (p[0] | p[1] << 8); }

// PCM (8/16/24/32-bit) and IEEE float (32/64-bit) WAV/BWF; downmix by arithmetic mean, as the
// reference processor does (wav.mean(dim=1)).
bool read_wav(const char* path, Wav& out, std::string& err) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { err = std::string("cannot open ") + path; return false; }
    std::vector<unsigned char> buf;
    {
        std::fseek(f, 0, SEEK_END);
        const long n = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        if (n <= 0) { std::fclose(f); err = "empty file " + std::string(path); return false; }
        buf.resize((size_t) n);
        if (std::fread(buf.data(), 1, buf.size(), f) != buf.size()) {
            std::fclose(f);
            err = "short read on " + std::string(path);
            return false;
        }
    }
    std::fclose(f);
    if (buf.size() < 44 || std::memcmp(buf.data(), "RIFF", 4) != 0 || std::memcmp(buf.data() + 8, "WAVE", 4) != 0) {
        err = std::string("not a RIFF/WAVE file: ") + path;
        return false;
    }
    int fmt = 0, channels = 0, bits = 0, sample_rate = 0;
    const unsigned char* data = nullptr;
    size_t data_bytes = 0;
    for (size_t at = 12; at + 8 <= buf.size();) {
        const uint32_t sz = rd_u32(&buf[at + 4]);
        const unsigned char* body = &buf[at + 8];
        if (std::memcmp(&buf[at], "fmt ", 4) == 0 && sz >= 16) {
            fmt = rd_u16(body);
            channels = rd_u16(body + 2);
            sample_rate = (int) rd_u32(body + 4);
            bits = rd_u16(body + 14);
            if (fmt == 0xFFFE && sz >= 40) fmt = rd_u16(body + 24);   // WAVE_FORMAT_EXTENSIBLE -> subformat
        } else if (std::memcmp(&buf[at], "data", 4) == 0) {
            data = body;
            data_bytes = std::min<size_t>(sz, buf.size() - (size_t) (body - buf.data()));
        }
        at += 8 + (size_t) sz + (sz & 1);
    }
    if (!data) { err = "no data chunk in " + std::string(path); return false; }
    if (channels <= 0 || sample_rate <= 0) { err = "bad fmt chunk in " + std::string(path); return false; }
    const bool is_float = fmt == 3;
    if (!is_float && fmt != 1) { err = "unsupported WAV format tag " + std::to_string(fmt); return false; }
    if (is_float && bits != 32 && bits != 64) { err = "unsupported float WAV width " + std::to_string(bits); return false; }
    if (!is_float && bits != 8 && bits != 16 && bits != 24 && bits != 32)
        { err = "unsupported PCM width " + std::to_string(bits); return false; }
    const int bytes = bits / 8;
    const int64_t frames = (int64_t) (data_bytes / ((size_t) bytes * (size_t) channels));
    if (frames == 0) { err = "no samples in " + std::string(path); return false; }
    out.samples.resize((size_t) frames);
    out.frames = frames;
    out.channels = channels;
    out.sample_rate = sample_rate;
    for (int64_t i = 0; i < frames; ++i) {
        double acc = 0.0;
        for (int c = 0; c < channels; ++c) {
            const unsigned char* p = data + ((size_t) i * channels + (size_t) c) * (size_t) bytes;
            double v = 0.0;
            if (is_float) {
                if (bits == 32) { float x; std::memcpy(&x, p, 4); v = x; }
                else { double x; std::memcpy(&x, p, 8); v = x; }
            } else if (bits == 16) {
                v = (double) (int16_t) rd_u16(p) / 32768.0;
            } else if (bits == 24) {
                int32_t s = (int32_t) ((uint32_t) p[0] << 8 | (uint32_t) p[1] << 16 | (uint32_t) p[2] << 24) >> 8;
                v = (double) s / 8388608.0;
            } else if (bits == 32) {
                v = (double) (int32_t) rd_u32(p) / 2147483648.0;
            } else {
                v = ((double) p[0] - 128.0) / 128.0;
            }
            acc += v;
        }
        out.samples[(size_t) i] = (float) (acc / channels);
    }
    return true;
}

// torchaudio.functional.resample's kernel: Hann-windowed sinc, lowpass_filter_width=6,
// rolloff=0.99, evaluated directly at each output time.  (torchaudio applies it as a polyphase
// conv1d; the values are the same, and this form has no phase bookkeeping to get wrong.)
//   y[m] = sum_i x[i] * sinc(base*(t - i/orig)) * cos^2(pi/2 * clamp(...)/6) * base/orig,  t = m/new
// with the rates reduced by their gcd and base = min(orig,new)*0.99.
std::vector<float> resample(const std::vector<float>& x, int orig_freq, int new_freq) {
    if (orig_freq == new_freq || x.empty()) return x;
    const int g = std::gcd(orig_freq, new_freq);
    const double of = (double) (orig_freq / g), nf = (double) (new_freq / g);
    const double base = std::min(of, nf) * 0.99;
    const int width = (int) std::ceil(6.0 * of / base) + 1;
    const int64_t n = (int64_t) x.size();
    const int64_t n_out = (int64_t) std::ceil(nf * (double) n / of);
    std::vector<float> y((size_t) n_out, 0.0f);
    const double scale = base / of;
    for (int64_t m = 0; m < n_out; ++m) {
        const double t = (double) m / nf;
        const int64_t i0 = (int64_t) std::floor(t * of);
        double acc = 0.0;
        for (int64_t i = i0 - width; i <= i0 + width; ++i) {
            if (i < 0 || i >= n) continue;
            double a = base * (t - (double) i / of);
            a = std::max(-6.0, std::min(6.0, a));
            const double cw = std::cos(a * M_PI / 12.0);
            const double w = cw * cw;
            const double tt = a * M_PI;
            const double sn = (tt == 0.0) ? 1.0 : std::sin(tt) / tt;
            acc += (double) x[(size_t) i] * sn * w * scale;
        }
        y[(size_t) m] = (float) acc;
    }
    return y;
}

// ---------------------------------------------------------------- log-mel front end

// torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=64, n_fft=400, win_length=400,
//     hop_length=160, center=True, pad_mode='reflect', power=2.0, norm=None, mel_scale='htk')
// then log(clamp(x, 1e-9, 1e9)); HTK mel points, no area normalization.
struct MelBank {
    std::vector<float> fb;             // [bins][n_mels]
    std::vector<double> window;        // periodic Hann, length n_fft
    std::vector<double> cos_t, sin_t;   // [n_fft][bins] (frame-major, for the accumulation loop)

    std::vector<float> shipped;         // the model file's table, copied out when the weights live on the GPU
    const float* shipped_fb = nullptr;   // [bin][mel] table from the model file, when present

    void init() {
        const int bins = (int) (kNfft / 2 + 1);
        const double sr = (double) kSampleRate;
        if (const float* shipped = shipped_fb) {
            fb.assign(shipped, shipped + (size_t) (kNfft / 2 + 1) * kNMel);
            build_window_and_tables();
            return;
        }
        auto hz_to_mel = [](double f) { return 2595.0 * std::log10(1.0 + f / 700.0); };
        auto mel_to_hz = [](double m) { return 700.0 * (std::pow(10.0, m / 2595.0) - 1.0); };
        const double m_min = hz_to_mel(0.0), m_max = hz_to_mel(sr / 2.0);
        std::vector<double> f_pts((size_t) kNMel + 2);
        for (int i = 0; i < kNMel + 2; ++i)
            f_pts[(size_t) i] = mel_to_hz(m_min + (m_max - m_min) * (double) i / (double) (kNMel + 1));
        fb.assign((size_t) bins * kNMel, 0.0f);
        for (int k = 0; k < bins; ++k) {
            const double f = (double) k * (sr / 2.0) / (double) (bins - 1);   // linspace(0, sr/2, bins)
            for (int m = 0; m < kNMel; ++m) {
                const double lo = f_pts[(size_t) m], mid = f_pts[(size_t) m + 1], hi = f_pts[(size_t) m + 2];
                const double down = (f - lo) / (mid - lo);     // torchaudio _create_triangular_filterbank
                const double up = (hi - f) / (hi - mid);
                fb[(size_t) k * kNMel + m] = (float) std::max(0.0, std::min(down, up));
            }
        }
        build_window_and_tables();
    }

    void build_window_and_tables() {
        const int bins = (int) (kNfft / 2 + 1);
        window.resize((size_t) kNfft);
        for (int n = 0; n < kNfft; ++n)   // torch.hann_window(400): periodic, so /N not /(N-1)
            window[(size_t) n] = 0.5 - 0.5 * std::cos(2.0 * M_PI * n / (double) kNfft);
        cos_t.assign((size_t) kNfft * bins, 0.0);
        sin_t.assign((size_t) kNfft * bins, 0.0);
        for (int n = 0; n < kNfft; ++n)
            for (int k = 0; k < bins; ++k) {
                const double a = 2.0 * M_PI * (double) k * (double) n / (double) kNfft;
                cos_t[(size_t) n * bins + k] = std::cos(a);
                sin_t[(size_t) n * bins + k] = std::sin(a);
            }
    }

    // out: [T_mel][kNMel]; T_mel = 1 + floor(n/160), center=True with reflect padding.
    // Frames are independent, so they are split across threads (the same per-frame arithmetic, bit for bit).
    void compute(const float* x, int64_t n, std::vector<float>& out, int64_t& t_mel) const {
        const int64_t pad = kNfft / 2;
        const int bins = (int) (kNfft / 2 + 1);
        t_mel = n > 0 ? 1 + n / kHop : 0;
        out.assign((size_t) t_mel * kNMel, 0.0f);
        if (const char* fb_path = std::getenv("STRATA_AUDIO_DUMP_FB")) {   // tests: the filterbank used
            static bool once = false;
            if (!once) {
                once = true;
                if (FILE* g = std::fopen(fb_path, "wb")) {
                    std::fwrite(fb.data(), 4, fb.size(), g);
                    std::fclose(g);
                }
            }
        }
        auto frames = [&](int64_t t0, int64_t t1) {
            std::vector<double> re((size_t) bins), im((size_t) bins);
            std::vector<double> frame((size_t) kNfft);
            for (int64_t t = t0; t < t1; ++t) {
                const int64_t start = t * kHop - pad;
                for (int64_t i = 0; i < kNfft; ++i) {
                    const int64_t s = start + i;
                    float v;
                    if (s >= 0 && s < n) v = x[s];
                    else if (s < 0) {
                        const int64_t r = -s;                       // reflect: -1 -> x[1]
                        v = (r < n) ? x[r] : 0.0f;
                    } else {
                        const int64_t r = 2 * (n - 1) - s;           // reflect: n -> x[n-2]
                        v = (r >= 0 && r < n) ? x[r] : 0.0f;
                    }
                    frame[(size_t) i] = (double) v * window[(size_t) i];   // f64: the f32 windowing was the
                                                                           // largest term left in the mel error
                }
                std::fill(re.begin(), re.end(), 0.0f);
                std::fill(im.begin(), im.end(), 0.0f);
                for (int64_t i = 0; i < kNfft; ++i) {
                    const double v = frame[(size_t) i];
                    if (v == 0.0) continue;
                    const double* c = &cos_t[(size_t) i * bins];
                    const double* s = &sin_t[(size_t) i * bins];
                    for (int k = 0; k < bins; ++k) {
                        re[(size_t) k] += v * c[k];
                        im[(size_t) k] += v * s[k];
                    }
                }
                float* dst = &out[(size_t) t * kNMel];
                double acc[kNMel] = {0.0};
                for (int k = 0; k < bins; ++k) {
                    const double p = re[(size_t) k] * re[(size_t) k] + im[(size_t) k] * im[(size_t) k];
                    const float* row = &fb[(size_t) k * kNMel];
                    for (int m = 0; m < kNMel; ++m)
                        if (row[m] != 0.0f) acc[m] += p * (double) row[m];
                }
                for (int m = 0; m < kNMel; ++m) {
                    const double v = std::min(std::max(acc[m], 1e-9), 1e9);
                    dst[m] = (float) std::log(v);
                }
            }
        };
        const int64_t nth = (int64_t) std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
        if (t_mel < 256 || nth <= 1) { frames(0, t_mel); return; }
        std::vector<std::thread> pool;
        const int64_t step = (t_mel + nth - 1) / nth;
        for (int64_t a = 0; a < t_mel; a += step)
            pool.emplace_back(frames, a, std::min(t_mel, a + step));
        for (auto& th : pool) th.join();
    }
};

// ---------------------------------------------------------------- ggml plumbing

struct Weights {
    ggml_context* ctx = nullptr;
    ggml_backend_t backend = nullptr;      // CPU: the whole CPU build, and the fallback of the CUDA one
    ggml_backend_t gpu = nullptr;          // CUDA backend, only with --gpu
    ggml_backend_sched_t sched = nullptr;  // CUDA + CPU; null on the CPU path
    ggml_gallocr_t galloc = nullptr;       // CPU path only
    bool on_gpu = false;
    int gpu_device = -1;
    std::unordered_map<std::string, ggml_tensor*> w;
    std::unordered_map<std::string, std::string> meta_s;

    ~Weights() {
        if (sched) ggml_backend_sched_free(sched);
        if (galloc) ggml_gallocr_free(galloc);
        if (gpu) ggml_backend_free(gpu);
        if (backend) ggml_backend_free(backend);
        if (ctx) ggml_free(ctx);
    }

    // A tensor's bytes, whatever its backend: a memcpy from the ggml arena on CPU, a device-to-host copy on CUDA.
    void read(const ggml_tensor* t, void* dst, size_t bytes) const {
        if (t->buffer) ggml_backend_tensor_get(t, dst, 0, bytes);
        else std::memcpy(dst, t->data, bytes);
    }

    bool load(const char* path, int threads, int device, std::string& err) {
        ggml_log_set(quiet_log, nullptr);
        ggml_backend_load_all();
        backend = ggml_backend_cpu_init();
        if (!backend) { err = "no CPU backend"; return false; }
        ggml_backend_cpu_set_n_threads(backend, threads);

        if (device >= 0) {
#ifdef STRATA_AUDIO_CUDA
            gpu = ggml_backend_cuda_init(device);
            if (!gpu) { err = "CUDA backend did not start on device " + std::to_string(device); return false; }
            on_gpu = true;
            gpu_device = device;
            ggml_backend_t backends[2] = {gpu, backend};      // the CPU backend must be last: op_offload's fallback
            sched = ggml_backend_sched_new(backends, nullptr, 2, 16384, false, true);
            if (!sched) { err = "cannot create the CUDA/CPU scheduler"; return false; }
#else
            err = "this is the CPU build of strata-audio; run engine-poly/strata-audio-cuda for --gpu";
            return false;
#endif
        } else {
            galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        }

        // No allocation in the gguf context on the GPU: the tensors are created as metadata only, allocated in one
        // CUDA buffer below, and filled from the file.  (On the CPU the old in-context allocation stays.)
        gguf_init_params pp{on_gpu, &ctx};
        gguf_context* gg = gguf_init_from_file(path, pp);
        if (!gg) { err = std::string("cannot read ") + path; return false; }
        const int64_t n = gguf_get_n_tensors(gg);
        for (int64_t i = 0; i < n; ++i) {
            const char* name = gguf_get_tensor_name(gg, i);
            ggml_tensor* t = ggml_get_tensor(ctx, name);
            if (!t) { err = std::string("tensor missing after load: ") + name; gguf_free(gg); return false; }
            w.emplace(name, t);
        }
        for (const char* key : {"gca.arch", "gca.quant", "general.architecture"}) {
            const int64_t k = gguf_find_key(gg, key);
            if (k >= 0) meta_s[key] = gguf_get_val_str(gg, k);
        }
        if (on_gpu) {
            ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(ctx, gpu);
            if (!wbuf) { err = "cannot allocate the encoder weights on the GPU"; gguf_free(gg); return false; }
            // Stream the tensor data from the file into the CUDA buffers in 4 MB pieces (no second full copy in RAM).
            FILE* f = std::fopen(path, "rb");
            if (!f) { err = std::string("cannot reopen ") + path; gguf_free(gg); return false; }
            std::vector<char> buf(4u << 20);
            const size_t data0 = gguf_get_data_offset(gg);
            for (int64_t i = 0; i < n && err.empty(); ++i) {
                ggml_tensor* t = ggml_get_tensor(ctx, gguf_get_tensor_name(gg, i));
                const size_t nbytes = ggml_nbytes(t);
                if (std::fseek(f, (long) (data0 + gguf_get_tensor_offset(gg, i)), SEEK_SET) != 0) {
                    err = "cannot seek in " + std::string(path);
                    break;
                }
                for (size_t done = 0; done < nbytes;) {
                    const size_t want = std::min(buf.size(), nbytes - done);
                    if (std::fread(buf.data(), 1, want, f) != want) {
                        err = "short read on " + std::string(path);
                        break;
                    }
                    ggml_backend_tensor_set(t, buf.data(), done, want);
                    done += want;
                }
            }
            std::fclose(f);
            if (!err.empty()) { gguf_free(gg); return false; }
        }
        gguf_free(gg);
        return true;
    }


    ggml_tensor* t(const std::string& name) const {
        auto it = w.find(name);
        if (it == w.end()) {
            std::fprintf(stderr, "strata-audio: missing tensor %s\n", name.c_str());
            std::abort();
        }
        return it->second;
    }
};

std::string wname(int layer, const char* suffix) {
    return "encoder.layers." + std::to_string(layer) + "." + suffix;
}

// ---------------------------------------------------------------- GPU sampling
//
// What the encoder actually used: NVML through dlopen, so the CPU build and a machine without the driver keep
// working (no link dependency, no headers).  One sample every 200 ms while an ENC runs: mean GPU utilization and
// the peak of this process's own VRAM (the decoder's ~6.5 GB on the same card is not counted).
struct GpuSample {
    double util = -1.0;     // mean GPU busy over the encode, percent; -1 = not sampled
    int vram_mib = -1;      // peak VRAM this process held, MiB
    int device_mib = -1;    // peak device-wide VRAM in use, MiB (what nvidia-smi's memory.used shows)
    bool on = false;
};

#ifdef STRATA_AUDIO_CUDA
class GpuMonitor {
public:
    ~GpuMonitor() { stop(); }

    bool start(int device) {
#ifdef _WIN32
        h_ = dlopen("nvml.dll", RTLD_LAZY);
        if (!h_) h_ = dlopen("C:/Program Files/NVIDIA Corporation/NVSMI/nvml.dll", RTLD_LAZY);
#else
        h_ = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
        if (!h_) h_ = dlopen("libnvidia-ml.so", RTLD_LAZY);
#endif
        if (!h_) return false;
        auto sym = [&](const char* n) { return dlsym(h_, n); };
        init_ = (decltype(init_)) sym("nvmlInit_v2");
        shutdown_ = (decltype(shutdown_)) sym("nvmlShutdown");
        handle_ = (decltype(handle_)) sym("nvmlDeviceGetHandleByIndex_v2");
        util_ = (decltype(util_)) sym("nvmlDeviceGetUtilizationRates");
        mem_ = (decltype(mem_)) sym("nvmlDeviceGetMemoryInfo");
        procs_ = (decltype(procs_)) sym("nvmlDeviceGetComputeRunningProcesses_v3");
        if (!procs_) procs_ = (decltype(procs_)) sym("nvmlDeviceGetComputeRunningProcesses_v2");
        if (!procs_) procs_ = (decltype(procs_)) sym("nvmlDeviceGetComputeRunningProcesses");
        if (!init_ || !handle_ || !util_ || !mem_) { dlclose(h_); h_ = nullptr; return false; }
        if (init_() != 0) { dlclose(h_); h_ = nullptr; return false; }
        if (handle_((unsigned) device, &dev_) != 0) { shutdown_(); dlclose(h_); h_ = nullptr; return false; }
        pid_ = (unsigned)
#ifdef _WIN32
            _getpid();
#else
            getpid();
#endif
        run_ = true;
        thread_ = std::thread(&GpuMonitor::loop, this);
        return true;
    }

    void begin() {
        std::lock_guard<std::mutex> g(mu_);
        util_sum_ = 0.0; util_n_ = 0; peak_self_ = 0; peak_dev_ = 0;
        self_ok_ = false;
    }

    GpuSample end() {
        stop();
        GpuSample s;
        std::lock_guard<std::mutex> g(mu_);
        if (util_n_) { s.util = util_sum_ / util_n_; s.on = true; }
        if (peak_self_ > 0) s.vram_mib = peak_self_;        // 0 = sampled before any allocation: unknown
        else if (peak_dev_ > 0) s.vram_mib = peak_dev_;     // no per-process API: the device's high-water mark
        if (peak_dev_ > 0) s.device_mib = peak_dev_;
        return s;
    }

    // What it holds right now, without stopping: the PROGRESS lines carry these.
    GpuSample peek() {
        GpuSample s;
        std::lock_guard<std::mutex> g(mu_);
        if (util_n_) { s.util = util_sum_ / util_n_; s.on = true; }
        s.vram_mib = (int) (peak_self_ > 0 ? peak_self_ : peak_dev_);
        s.device_mib = (int) peak_dev_;
        return s;
    }

private:
    using nvmlDevice_t = void*;
    struct Util { unsigned gpu; unsigned memory; };
    struct Mem { unsigned long long total, free, used; unsigned long long reserved; };
    struct Proc { unsigned pid; unsigned long long used; unsigned instance; unsigned compute_instance; };

    void loop() {
        while (run_) {
            unsigned u = 0;
            Util x{};
            if (util_ && dev_ && util_(dev_, &x) == 0) u = x.gpu;
            unsigned long long dev_used = 0, self_used = 0;
            bool self_known = false;
            if (mem_) { Mem m{}; if (mem_(dev_, &m) == 0) dev_used = m.used; }
            if (procs_) {
                Proc p[64]; unsigned n = 64;
                if (procs_(dev_, &n, p) == 0) {
                    for (unsigned i = 0; i < n && i < 64; ++i)
                        if (p[i].pid == pid_) { self_used = p[i].used; self_known = true; break; }
                }
            }
            {
                std::lock_guard<std::mutex> g(mu_);
                if (u) { util_sum_ += u; util_n_ += 1; }
                if (self_known) { self_ok_ = true; if ((long long) (self_used >> 20) > peak_self_) peak_self_ = self_used >> 20; }
                if ((long long) (dev_used >> 20) > peak_dev_) peak_dev_ = dev_used >> 20;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    }

    void stop() {
        if (!run_.exchange(false)) return;
        if (thread_.joinable()) thread_.join();
        if (shutdown_) shutdown_();
        if (h_) { dlclose(h_); h_ = nullptr; }
    }

    void* h_ = nullptr;
    void* dev_ = nullptr;
    unsigned pid_ = 0;
    std::atomic<bool> run_{false};
    std::thread thread_;
    std::mutex mu_;
    double util_sum_ = 0.0;
    int util_n_ = 0;
    long long peak_self_ = -1;
    long long peak_dev_ = -1;
    bool self_ok_ = false;
    int (*init_)(void) = nullptr;
    int (*shutdown_)(void) = nullptr;
    int (*handle_)(unsigned, void**) = nullptr;
    int (*util_)(void*, Util*) = nullptr;
    int (*mem_)(void*, Mem*) = nullptr;
    int (*procs_)(void*, unsigned*, Proc*) = nullptr;
};
#else
struct GpuMonitor {
    bool start(int) { return false; }
    void begin() {}
    GpuSample peek() { return {}; }
    GpuSample end() { return {}; }
};
#endif


// ---------------------------------------------------------------- encoder

struct AudioEncoder {
    Weights wt;
    MelBank mel;
    int64_t group_blocks = 8;      // 200-frame blocks per graph (0 = whole file in one graph)
    int n_layers_dbg = 24;         // debug: run only the first N Conformer layers
    int64_t n_embd = 1536;
    std::string dump_dir;          // when set, writes mel/pre_encode/layer_NN/rows as raw f32
    GpuMonitor* gpu = nullptr;     // sampler while an ENC runs (null on the CPU build)
    bool progress = false;         // emit PROGRESS lines (after READY; never during the warm-up)
    struct Phases { double mel_ms = 0, pre_ms = 0, win_ms = 0, adapter_ms = 0; };
    Phases phases;
    int64_t chunks = 0;            // graph executions (window passes) of the last encode

    // Where the graph's working buffers go: the scheduler's when the weights are on the GPU (CUDA + CPU
    // fallback), the CPU gallocr otherwise.  `persistent` reuses wt.galloc for the small graphs.
    ggml_gallocr_t alloc(ggml_cgraph* gf, bool persistent = false) {
        if (wt.sched) {
            ggml_backend_sched_reset(wt.sched);
            if (!ggml_backend_sched_alloc_graph(wt.sched, gf)) {
                std::fprintf(stderr, "strata-audio: scheduler allocation failed\n");
                std::abort();
            }
            return nullptr;
        }
        ggml_gallocr_t gal = persistent ? wt.galloc
                                        : ggml_gallocr_new(ggml_backend_get_default_buffer_type(wt.backend));
        if (!ggml_gallocr_alloc_graph(gal, gf)) {
            std::fprintf(stderr, "strata-audio: graph allocation failed\n");
            std::abort();
        }
        return gal;
    }

    void compute(ggml_cgraph* gf) {
        const ggml_status st = wt.sched ? ggml_backend_sched_graph_compute(wt.sched, gf)
                                        : ggml_backend_graph_compute(wt.backend, gf);
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "strata-audio: graph compute failed (%d)\n", (int) st);
            std::abort();
        }
    }

    // ---- one 24-layer pass over `tw` pre-encoded frames starting at global frame `f0`.
    // x: [kDModel][tw] channel-major; out gets the same layout (all 24 layers applied).
    // layer_dumps[i] receives layer i's output when non-empty.
    void run_window(const float* xin, int64_t tw, int64_t f0, const int64_t total_frames,
                    std::vector<float>& out, std::vector<std::vector<float>>* layer_dumps) {
        ggml_init_params ip{48u << 20, nullptr, true};
        ggml_context* ctx = ggml_init(ip);
        ggml_cgraph* gf = ggml_new_graph_custom(ctx, 8192, false);

        ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kDModel, tw);
        ggml_set_name(x, "input");
        ggml_set_input(x);
        ggml_tensor* x_in = x;

        // chunk-rotation positions: the phase restarts every 200 frames
        ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tw);
        ggml_set_name(pos, "pos");
        ggml_set_input(pos);

        // block-diagonal, non-causal mask: 0 inside the same 200-frame block, -inf everywhere else
        ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, tw, tw);
        ggml_set_name(mask, "mask");
        ggml_set_input(mask);
        std::vector<ggml_tensor*> layer_out;
        std::vector<ggml_tensor*> stage_out;

        auto heads3 = [&](ggml_tensor* a) {
            return ggml_view_3d(ctx, a, kHeadDim, kHeads, tw, kHeadDim * 4, kDModel * 4, 0);
        };
        auto lin = [&](ggml_tensor* a, ggml_tensor* W, ggml_tensor* b) {
            ggml_tensor* y = ggml_mul_mat(ctx, W, a);
            return b ? ggml_add(ctx, y, b) : y;
        };
        auto ln = [&](ggml_tensor* a, ggml_tensor* w, ggml_tensor* b) {
            ggml_tensor* y = ggml_norm(ctx, a, kNormEps);
            if (w) y = ggml_mul(ctx, y, w);
            if (b) y = ggml_add(ctx, y, b);
            return y;
        };
        auto ffn = [&](ggml_tensor* a, const std::string& p) {
            ggml_tensor* h = lin(a, wt.t(p + ".linear1.weight"), wt.t(p + ".linear1.bias"));
            h = ggml_silu(ctx, h);
            return lin(h, wt.t(p + ".linear2.weight"), wt.t(p + ".linear2.bias"));
        };

        const float attn_scale = 1.0f / std::sqrt((float) kHeadDim);

        for (int L = 0; L < n_layers_dbg; ++L) {
            // x += 0.5 * FFN1(LN1(x))
            ggml_tensor* n1 = ln(x, wt.t(wname(L, "norm_feed_forward1.weight")), wt.t(wname(L, "norm_feed_forward1.bias")));
            ggml_tensor* h1 = ffn(n1, wname(L, "feed_forward1"));
            if (layer_dumps && L == 0 && !std::getenv("STRATA_AUDIO_NOSTAGE")) { ggml_set_output(h1); ggml_set_name(h1, "s0_ffn1"); stage_out.push_back(h1); }
            if (layer_dumps && L == 0 && !std::getenv("STRATA_AUDIO_NOSTAGE")) { ggml_set_output(n1); ggml_set_name(n1, "s0_n1"); stage_out.push_back(n1); }
            x = ggml_add(ctx, x, ggml_scale(ctx, h1, 0.5f));

            // x += Attn(LN2(x));  RoPE on the LayerNorm output, before the Q/K projections
            ggml_tensor* n2 = ln(x, wt.t(wname(L, "norm_self_att.weight")), wt.t(wname(L, "norm_self_att.bias")));
            ggml_tensor* rot = ggml_rope_ext(ctx, heads3(n2), pos, nullptr, (int) kHeadDim,
                                             GGML_ROPE_TYPE_NEOX, 131072, kRopeBase, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
            ggml_tensor* xr = ggml_reshape_2d(ctx, ggml_cont(ctx, rot), kDModel, tw);
            ggml_tensor* q = lin(xr, wt.t(wname(L, "self_attn.linear_q.weight")), wt.t(wname(L, "self_attn.linear_q.bias")));
            ggml_tensor* k = lin(xr, wt.t(wname(L, "self_attn.linear_k.weight")), wt.t(wname(L, "self_attn.linear_k.bias")));
            ggml_tensor* v = lin(n2, wt.t(wname(L, "self_attn.linear_v.weight")), wt.t(wname(L, "self_attn.linear_v.bias")));
            ggml_tensor* qp = ggml_cont(ctx, ggml_permute(ctx, heads3(q), 0, 2, 1, 3));   // [64, tw, 16]
            ggml_tensor* kp = ggml_cont(ctx, ggml_permute(ctx, heads3(k), 0, 2, 1, 3));
            ggml_tensor* vp = ggml_cont(ctx, ggml_permute(ctx, heads3(v), 1, 2, 0, 3));   // [tw, 64, 16]
            ggml_tensor* s = ggml_mul_mat(ctx, kp, qp);                                   // [tw, tw, 16]
            ggml_tensor* p = ggml_soft_max_ext(ctx, s, mask, attn_scale, 0.0f);
            ggml_tensor* o = ggml_mul_mat(ctx, p, vp);                                    // [tw, 64, 16]
            ggml_tensor* o2 = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, o, 2, 0, 1, 3)), kDModel, tw);
            ggml_tensor* ao = lin(o2, wt.t(wname(L, "self_attn.linear_out.weight")), wt.t(wname(L, "self_attn.linear_out.bias")));
            if (layer_dumps && L == 0 && !std::getenv("STRATA_AUDIO_NOSTAGE")) { ggml_set_output(ao); ggml_set_name(ao, "s0_attn"); stage_out.push_back(ao); }
            if (layer_dumps && L == 0 && !std::getenv("STRATA_AUDIO_NOSTAGE")) { ggml_set_output(o2); ggml_set_name(o2, "s0_attn_ctx"); stage_out.push_back(o2); }
            x = ggml_add(ctx, x, ao);

            // x += Conv(LN3(x))
            ggml_tensor* n3 = ln(x, wt.t(wname(L, "norm_conv.weight")), wt.t(wname(L, "norm_conv.bias")));
            ggml_tensor* pw = lin(n3, wt.t(wname(L, "conv.pointwise_conv1.weight")), wt.t(wname(L, "conv.pointwise_conv1.bias")));
            ggml_tensor* ga = ggml_cont(ctx, ggml_view_2d(ctx, pw, kDModel, tw, 2 * kDModel * 4, 0));
            ggml_tensor* gb = ggml_cont(ctx, ggml_view_2d(ctx, pw, kDModel, tw, 2 * kDModel * 4, (size_t) kDModel * 4));
            ggml_tensor* gl = ggml_mul(ctx, ga, ggml_sigmoid(ctx, gb));
            // depthwise Conv1d(k=5, groups=1024, pad=2) as 5 shifted taps on a zero-padded copy
            // zero pad 2 frames on each side: ggml_pad() only pads on the right, ggml_pad_ext(l,r,...) does both
            ggml_tensor* pad2 = ggml_pad_ext(ctx, gl, 0, 0, 2, 2, 0, 0, 0, 0);              // [1024, tw+4]
            ggml_tensor* dw = nullptr;
            for (int k = 0; k < (int) kConvK; ++k) {
                ggml_tensor* tap = ggml_view_2d(ctx, pad2, kDModel, tw, kDModel * 4, (size_t) k * kDModel * 4);
                ggml_tensor* wk = ggml_view_1d(ctx, wt.t(wname(L, "conv.depthwise_conv.weight")), kDModel,
                                               (size_t) k * kDModel * 4);
                ggml_tensor* t = ggml_mul(ctx, tap, wk);
                dw = dw ? ggml_add(ctx, dw, t) : t;
            }
            dw = ggml_add(ctx, dw, wt.t(wname(L, "conv.depthwise_conv.bias")));
            dw = ggml_silu(ctx, ln(dw, wt.t(wname(L, "conv.batch_norm.weight")), wt.t(wname(L, "conv.batch_norm.bias"))));
            ggml_tensor* cv = lin(dw, wt.t(wname(L, "conv.pointwise_conv2.weight")), wt.t(wname(L, "conv.pointwise_conv2.bias")));
            if (layer_dumps && L == 0 && !std::getenv("STRATA_AUDIO_NOSTAGE")) { ggml_set_output(cv); ggml_set_name(cv, "s0_conv"); stage_out.push_back(cv); }
            if (layer_dumps && L == 0 && !std::getenv("STRATA_AUDIO_NOSTAGE")) { ggml_set_output(dw); ggml_set_name(dw, "s0_dw"); stage_out.push_back(dw); }
            x = ggml_add(ctx, x, cv);

            // x += 0.5 * FFN2(LN4(x)); x = LN5(x)
            ggml_tensor* n4 = ln(x, wt.t(wname(L, "norm_feed_forward2.weight")), wt.t(wname(L, "norm_feed_forward2.bias")));
            ggml_tensor* h2 = ffn(n4, wname(L, "feed_forward2"));
            if (layer_dumps && L == 0 && !std::getenv("STRATA_AUDIO_NOSTAGE")) { ggml_set_output(h2); ggml_set_name(h2, "s0_ffn2"); stage_out.push_back(h2); }
            x = ggml_add(ctx, x, ggml_scale(ctx, h2, 0.5f));
            x = ln(x, wt.t(wname(L, "norm_out.weight")), wt.t(wname(L, "norm_out.bias")));

            if (layer_dumps) {
                if (!std::getenv("STRATA_AUDIO_NOSTAGE")) { ggml_set_output(x); ggml_set_name(x, ("layer" + std::to_string(L)).c_str()); }
                layer_out.push_back(x);
            }
        }
        ggml_set_output(x);
        ggml_build_forward_expand(gf, x);
        ggml_gallocr_t gal = alloc(gf);
        ggml_backend_tensor_set(x_in, xin, 0, (size_t) kDModel * tw * 4);
        {
            std::vector<int32_t> p((size_t) tw);
            for (int64_t i = 0; i < tw; ++i) p[(size_t) i] = (int32_t) ((f0 + i) % kChunk);
            ggml_backend_tensor_set(pos, p.data(), 0, p.size() * 4);
        }
        {
            std::vector<float> m((size_t) (tw * tw), -INFINITY);
            for (int64_t j = 0; j < tw; ++j) {           // query
                const int64_t bj = (f0 + j) / kChunk;
                for (int64_t i = 0; i < tw; ++i) {       // key
                    const int64_t ti = f0 + i;
                    if (ti >= total_frames) continue;
                    if (ti / kChunk == bj) m[(size_t) (j * tw + i)] = 0.0f;
                }
            }
            ggml_backend_tensor_set(mask, m.data(), 0, m.size() * 4);
        }
        compute(gf);
        out.resize((size_t) kDModel * tw);
        wt.read(x, out.data(), out.size() * 4);
        if (!stage_out.empty()) {
            std::vector<float> tmp((size_t) kDModel * tw);
            for (ggml_tensor* t : stage_out) {
                FILE* f = std::fopen((dump_dir + "/" + t->name + ".f32").c_str(), "wb");
                if (f) { wt.read(t, tmp.data(), tmp.size() * 4); std::fwrite(tmp.data(), 4, tmp.size(), f); std::fclose(f); }
            }
        }
        if (layer_dumps) {
            for (int L = 0; L < (int) layer_out.size(); ++L) {
                (*layer_dumps)[(size_t) L].resize((size_t) kDModel * tw);
                wt.read(layer_out[(size_t) L], (*layer_dumps)[(size_t) L].data(),
                        (*layer_dumps)[(size_t) L].size() * 4);
            }
        }
        if (gal) ggml_gallocr_free(gal);          // after the reads: its buffers hold the results
        ggml_free(ctx);
    }

    // mel (T_mel,64) -> pre-encoded [1024, T_enc]
    void pre_encode(const std::vector<float>& melbuf, int64_t t_mel, std::vector<float>& out, int64_t& t_enc) {
        const int64_t t1 = t_mel > 0 ? (t_mel - 1) / 2 + 1 : 0;
        t_enc = t1 > 0 ? (t1 - 1) / 2 + 1 : 0;
        out.assign((size_t) (kDModel * std::max<int64_t>(t_enc, 0)), 0.0f);
        if (t_enc == 0) return;
        const int64_t chunk = 512;
        for (int64_t u0 = 0; u0 < t_enc; u0 += chunk) {
            const int64_t u1 = std::min(t_enc, u0 + chunk);
            const int64_t r0 = std::max<int64_t>(0, 2 * u0 - 2), r1 = std::min<int64_t>(t1, 2 * u1 + 2);
            const int64_t nr = r1 - r0;
            // im2col layout == the ggml tensor layout: ne0 (k*in + c) fastest, so frame-major rows
            std::vector<float> im1((size_t) (kConvK * kFeatIn) * (size_t) nr, 0.0f);
            for (int64_t r = 0; r < nr; ++r) {
                const int64_t centre = (r + r0) * 2 - 2;
                for (int64_t k = 0; k < kConvK; ++k) {
                    const int64_t s = centre + k;
                    if (s < 0 || s >= t_mel) continue;
                    for (int64_t c = 0; c < kFeatIn; ++c)
                        im1[(size_t) (r * (kConvK * kFeatIn) + k * kFeatIn + c)] = melbuf[(size_t) (s * kFeatIn + c)];
                }
            }
            std::vector<float> r1buf((size_t) (kDModel * nr), 0.0f);
            {
                ggml_init_params ip{16u << 20, nullptr, true};
                ggml_context* ctx = ggml_init(ip);
                ggml_cgraph* gf = ggml_new_graph(ctx);
                ggml_tensor* im = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kConvK * kFeatIn, nr);
                ggml_set_name(im, "im");
                ggml_tensor* y = ggml_mul_mat(ctx, wt.t("encoder.pre_encode.conv.0.weight"), im);
                y = ggml_add(ctx, y, wt.t("encoder.pre_encode.conv.0.bias"));
                ggml_set_output(y);
                ggml_build_forward_expand(gf, y);
                ggml_gallocr_t gal = alloc(gf, true);
                ggml_backend_tensor_set(im, im1.data(), 0, im1.size() * 4);
                compute(gf);
                wt.read(y, r1buf.data(), r1buf.size() * 4);
                ggml_free(ctx);
            }
            for (auto& v : r1buf) v = std::max(v, 0.0f);
            const int64_t nu = u1 - u0;
            std::vector<float> im2((size_t) (kConvK * kDModel) * (size_t) nu, 0.0f);
            for (int64_t u = 0; u < nu; ++u) {
                const int64_t centre = (u + u0) * 2 - 2;
                for (int64_t k = 0; k < kConvK; ++k) {
                    const int64_t s = centre + k;
                    if (s < r0 || s >= r1) continue;
                    std::memcpy(&im2[(size_t) (u * (kConvK * kDModel) + k * kDModel)],
                                &r1buf[(size_t) (s - r0) * kDModel], (size_t) kDModel * 4);
                }
            }
            {
                ggml_init_params ip{16u << 20, nullptr, true};
                ggml_context* ctx = ggml_init(ip);
                ggml_cgraph* gf = ggml_new_graph(ctx);
                ggml_tensor* im = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kConvK * kDModel, nu);
                ggml_set_name(im, "im");
                ggml_tensor* y = ggml_mul_mat(ctx, wt.t("encoder.pre_encode.conv.2.weight"), im);
                y = ggml_add(ctx, y, wt.t("encoder.pre_encode.conv.2.bias"));
                ggml_set_output(y);
                ggml_build_forward_expand(gf, y);
                ggml_gallocr_t gal = alloc(gf, true);
                ggml_backend_tensor_set(im, im2.data(), 0, im2.size() * 4);
                compute(gf);
                wt.read(y, &out[(size_t) u0 * kDModel], (size_t) nu * kDModel * 4);
                ggml_free(ctx);
            }
        }
        for (auto& v : out) v = std::max(v, 0.0f);
    }

    // rows = projector(stack(x)): x [1024][nf] channel-major -> out [L][1536], L = ceil(nf/4)
    void adapter(const float* x, int64_t nf, std::vector<float>& out) {
        const int64_t L = (nf + kStack - 1) / kStack;   // ceil: the reference pads the last row with zeros
        // ggml layout: [ne0 = k*1024 + c][ne1 = row], so row l holds x[4l..4l+3] back to back
        std::vector<float> buf((size_t) (kDModel * kStack) * (size_t) L, 0.0f);
        for (int64_t l = 0; l < L; ++l) {
            for (int64_t k = 0; k < kStack; ++k) {
                const int64_t f = l * kStack + k;
                if (f >= nf) continue;
                std::memcpy(&buf[(size_t) (l * (kDModel * kStack) + k * kDModel)],
                            &x[(size_t) f * kDModel], (size_t) kDModel * 4);
            }
        }
        ggml_init_params ip{16u << 20, nullptr, true};
        ggml_context* ctx = ggml_init(ip);
        ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_tensor* im = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kDModel * kStack, L);
        ggml_set_name(im, "stack");
        ggml_tensor* y = ggml_mul_mat(ctx, wt.t("modality_adapter.projector.linear.weight"), im);
        y = ggml_add(ctx, y, wt.t("modality_adapter.projector.linear.bias"));
        ggml_set_output(y);
        ggml_build_forward_expand(gf, y);
        ggml_gallocr_t gal = alloc(gf, true);
        ggml_backend_tensor_set(im, buf.data(), 0, buf.size() * 4);
        compute(gf);
        out.resize((size_t) (n_embd * L));
        wt.read(y, out.data(), out.size() * 4);
        ggml_free(ctx);
    }

    // whole file: wav samples -> rows [L][1536]
    void encode(const float* wav, int64_t n, std::vector<float>& rows, int64_t& t_mel, int64_t& t_enc) {
        auto clock = [] { return std::chrono::steady_clock::now(); };
        auto ms_since = [](auto t0) {
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        };
        phases = Phases{};
        chunks = 0;
        std::vector<float> melbuf;
        auto t_mel0 = clock();
        mel.compute(wav, n, melbuf, t_mel);
        phases.mel_ms = ms_since(t_mel0);
        std::vector<float> pre;
        auto t_pre0 = clock();
        pre_encode(melbuf, t_mel, pre, t_enc);
        phases.pre_ms = ms_since(t_pre0);
        if (!dump_dir.empty()) {
            FILE* f = std::fopen((dump_dir + "/mel.f32").c_str(), "wb");
            if (f) { std::fwrite(melbuf.data(), 4, melbuf.size(), f); std::fclose(f); }
            f = std::fopen((dump_dir + "/pre_encode.f32").c_str(), "wb");
            if (f) { std::fwrite(pre.data(), 4, pre.size(), f); std::fclose(f); }
        }
        const int64_t n_rows = (t_enc + kStack - 1) / kStack;
        rows.assign((size_t) (n_rows * n_embd), 0.0f);
        if (progress) {                                  // rows_total from the start, so the bar has a denominator
            std::printf("PROGRESS 0 %lld 0\n", (long long) n_rows);
            std::fflush(stdout);
        }
        const auto t_all = clock();

        int64_t blocks = (t_enc + kChunk - 1) / kChunk;
        int64_t gb = group_blocks > 0 ? group_blocks : blocks;
        for (int64_t b0 = 0; b0 < blocks; b0 += gb) {
            const int64_t b1 = std::min(blocks, b0 + gb);
            int64_t g0 = b0 * kChunk, g1 = std::min(t_enc, b1 * kChunk);
            int64_t w0 = std::max<int64_t>(0, (b0 - 1) * kChunk);   // one full block of halo each side:
            int64_t w1 = std::min<int64_t>(t_enc, (b1 + 1) * kChunk); // the conv needs 2 frames per layer
            std::vector<float> win((size_t) (kDModel * (w1 - w0)));
            std::memcpy(win.data(), &pre[(size_t) w0 * kDModel], win.size() * 4);
            std::vector<float> wout;
            std::vector<std::vector<float>> dumps;
            if (!dump_dir.empty() && b0 == 0 && b1 == blocks) dumps.resize(24);   // whole file in one window
            run_window(win.data(), w1 - w0, w0, t_enc, wout, dumps.empty() ? nullptr : &dumps);
            chunks += 1;
            if (!dumps.empty()) {
                for (int L = 0; L < n_layers_dbg; ++L) {
                    FILE* f = std::fopen((dump_dir + "/layer_" + std::to_string(L) + ".f32").c_str(), "wb");
                    if (f) { std::fwrite(dumps[(size_t) L].data(), 4, dumps[(size_t) L].size(), f); std::fclose(f); }
                }
            }
            // keep the group's own frames (the halo ones above/below are thrown away)
            std::vector<float> inner((size_t) (kDModel * (g1 - g0)));
            std::memcpy(inner.data(), &wout[(size_t) (g0 - w0) * kDModel], inner.size() * 4);
            if (!dump_dir.empty()) {
                FILE* f = std::fopen((dump_dir + "/encout_" + std::to_string(g0) + ".f32").c_str(), "wb");
                if (f) { std::fwrite(inner.data(), 4, inner.size(), f); std::fclose(f); }
            }
            std::vector<float> r;
            auto t_ad0 = clock();
            adapter(inner.data(), g1 - g0, r);
            phases.adapter_ms += ms_since(t_ad0);
            const int64_t l0 = g0 / kStack, nl = (int64_t) (r.size() / n_embd);
            std::memcpy(&rows[(size_t) l0 * n_embd], r.data(), (size_t) nl * n_embd * 4);
            if (progress) {                                  // what the server shows while the clip is encoded
                GpuSample s = gpu ? gpu->peek() : GpuSample{};
                char extra[96] = "";
                if (s.util >= 0) std::snprintf(extra, sizeof extra, " gpu_util=%d vram_mib=%d", (int) (s.util + 0.5), s.vram_mib);
                std::printf("PROGRESS %lld %lld %.0f%s\n", (long long) (g1 / kStack), (long long) n_rows,
                            ms_since(t_all), extra);
                std::fflush(stdout);
            }
        }
        phases.win_ms = ms_since(t_all) - phases.adapter_ms;
        if (!dump_dir.empty()) {
            FILE* f = std::fopen((dump_dir + "/rows.f32").c_str(), "wb");
            if (f) { std::fwrite(rows.data(), 4, rows.size(), f); std::fclose(f); }
        }
    }
};

// ---------------------------------------------------------------- process protocol

bool parse_enc(const std::string& line, std::string& wav, std::string& out) {
    if (line.rfind("ENC ", 0) != 0) return false;
    const std::string rest = line.substr(4);
    const size_t sp = rest.rfind(' ');
    if (sp == std::string::npos || sp == 0) return false;
    wav = rest.substr(0, sp);
    out = rest.substr(sp + 1);
    return !wav.empty() && !out.empty();
}

}  // namespace

int main(int argc, char** argv) {
    std::string model, dump_dir;
    int threads = 0;
    int64_t group_blocks = 8;
    int device = -1;                 // -1 = CPU; --gpu picks 0, --device N picks N
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--model") model = next();
        else if (a == "--threads") threads = std::atoi(next().c_str());
        else if (a == "--group-blocks") group_blocks = std::atoll(next().c_str());
        else if (a == "--dump-dir") dump_dir = next();
        else if (a == "--gpu") { if (device < 0) device = 0; }
        else if (a == "--device") device = std::atoi(next().c_str());
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (model.empty()) {
        std::fprintf(stderr, "usage: strata-audio --model <encoder.gguf> [--threads N] "
                             "[--group-blocks N] [--dump-dir DIR] [--gpu] [--device N]\n");
        return 2;
    }
    // CPU path (the default, and the only mode of the CPU build): keep the busy GPU unseen, as strata-vision
    // does.  With --gpu the card must stay visible to ggml; the server already gave this process the right
    // CUDA_VISIBLE_DEVICES (serve/audio.py).
    if (device < 0) {
#ifdef _WIN32
        _putenv_s("CUDA_VISIBLE_DEVICES", "-1");
#else
        setenv("CUDA_VISIBLE_DEVICES", "-1", 1);
#endif
    }
    if (threads <= 0) threads = (int) std::max(1u, std::thread::hardware_concurrency());
    if (!dump_dir.empty()) dump_dir.erase(dump_dir.find_last_not_of('/') + 1);

    AudioEncoder enc;
    enc.group_blocks = group_blocks;
    if (const char* dl = std::getenv("STRATA_AUDIO_LAYERS")) enc.n_layers_dbg = std::atoi(dl);
    enc.dump_dir = dump_dir;
    const auto t0 = std::chrono::steady_clock::now();
    std::string err;
    if (!enc.wt.load(model.c_str(), threads, device, err)) {
        std::printf("ERR %s\n", err.c_str());
        std::fflush(stdout);
        return 1;
    }
    if (enc.wt.w.count("audio.mel_fb")) {
        ggml_tensor* fb = enc.wt.t("audio.mel_fb");           // [bin][mel], from the model file
        enc.mel.shipped.resize((size_t) ggml_nelements(fb));
        enc.wt.read(fb, enc.mel.shipped.data(), enc.mel.shipped.size() * 4);
        enc.mel.shipped_fb = enc.mel.shipped.data();
    }
    enc.mel.init();
    std::fprintf(stderr, "strata-audio: weights loaded in %.1f s (%d threads, %s)\n",
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), threads,
                 device >= 0 ? ("CUDA device " + std::to_string(device)).c_str() : "CPU");
    size_t weights_bytes = 0;
    for (const auto& kv : enc.wt.w) weights_bytes += ggml_nbytes(kv.second);
    GpuMonitor monitor;
    if (device >= 0) {
        if (monitor.start(device)) enc.gpu = &monitor;
        else std::fprintf(stderr, "strata-audio: NVML is not available, so no GPU utilization/VRAM is reported\n");
    }
    // Warm up on one second of a 220 Hz tone: the first graph allocates its working buffers here,
    // not on the user's first request (as strata-vision does with its largest picture).
    {
        const auto wt0 = std::chrono::steady_clock::now();
        std::vector<float> tone(kSampleRate);
        for (int64_t i = 0; i < (int64_t) tone.size(); ++i)
            tone[(size_t) i] = 0.25f * std::sin((float) (2.0 * M_PI * 220.0 * (double) i / kSampleRate));
        std::vector<float> rows;
        int64_t tm = 0, te = 0;
        const std::string save_dump = enc.dump_dir;
        enc.dump_dir.clear();
        enc.encode(tone.data(), (int64_t) tone.size(), rows, tm, te);
        enc.dump_dir = save_dump;
        std::fprintf(stderr, "strata-audio: warmed up on 1.0 s in %.1f s (%lld mel frames, %lld rows)\n",
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - wt0).count(),
                     (long long) tm, (long long) (rows.size() / enc.n_embd));
    }
    std::printf("READY %lld\n", (long long) enc.n_embd);
    std::fflush(stdout);
    enc.progress = true;                 // from here the server reads our lines: PROGRESS is understood

    std::string line;
    while (std::getline(std::cin, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (line == "QUIT") break;
        std::string wav_path, out_path;
        if (!parse_enc(line, wav_path, out_path)) {
            std::printf("ERR expected: ENC <wav> <output>\n");
            std::fflush(stdout);
            continue;
        }
        const auto t_enc0 = std::chrono::steady_clock::now();
        Wav wav;
        if (!read_wav(wav_path.c_str(), wav, err)) {
            std::printf("ERR %s\n", err.c_str());
            std::fflush(stdout);
            continue;
        }
        std::vector<float> pcm = wav.sample_rate == kSampleRate ? wav.samples
                                                                : resample(wav.samples, wav.sample_rate, kSampleRate);
        if (pcm.empty()) {
            std::printf("ERR no samples after resampling %s\n", wav_path.c_str());
            std::fflush(stdout);
            continue;
        }
        if (const char* dump_pcm = std::getenv("STRATA_AUDIO_DUMP_PCM")) {   // tests: the downmixed, resampled PCM
            FILE* fp = std::fopen(dump_pcm, "wb");
            if (fp) { std::fwrite(pcm.data(), 4, pcm.size(), fp); std::fclose(fp); }
        }
        std::vector<float> rows;
        int64_t t_mel = 0, t_enc = 0;
        if (enc.gpu) enc.gpu->begin();
        enc.encode(pcm.data(), (int64_t) pcm.size(), rows, t_mel, t_enc);
        const GpuSample gs = enc.gpu ? enc.gpu->end() : GpuSample{};
        const int64_t n_rows = (int64_t) (rows.size() / enc.n_embd);
        FILE* f = std::fopen(out_path.c_str(), "wb");
        if (!f) {
            std::printf("ERR cannot write %s\n", out_path.c_str());
            std::fflush(stdout);
            continue;
        }
        const int32_t hdr[5] = {kSve1Magic, (int32_t) n_rows, (int32_t) n_rows, 1, (int32_t) enc.n_embd};
        bool ok = std::fwrite(hdr, sizeof hdr, 1, f) == 1;
        if (ok && n_rows) ok = std::fwrite(rows.data(), sizeof(float), rows.size(), f) == rows.size();
        std::fclose(f);
        if (!ok) {
            std::printf("ERR short write to %s\n", out_path.c_str());
            std::fflush(stdout);
            continue;
        }
        const double ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_enc0).count();
        const int64_t dur_ms = (int64_t) ((double) wav.frames * 1000.0 / wav.sample_rate);
        // The telemetry the server keeps: the device, the sampled utilization and VRAM, the graph executions, and
        // where the time went.  Values that were not sampled are left out (the CPU path has no GPU fields).
        std::string tail;
        {
            char b[512];
            if (device >= 0) std::snprintf(b, sizeof b, " device=cuda:%d", device);
            else std::snprintf(b, sizeof b, " device=cpu");
            tail += b;
            if (gs.util >= 0) {
                std::snprintf(b, sizeof b, " gpu_util=%d", (int) (gs.util + 0.5));
                tail += b;
            }
            if (gs.vram_mib >= 0) {
                std::snprintf(b, sizeof b, " vram_mib=%d", gs.vram_mib);
                tail += b;
            }
            if (gs.device_mib >= 0) {
                std::snprintf(b, sizeof b, " device_vram_mib=%d", gs.device_mib);
                tail += b;
            }
            if (gs.vram_mib >= 0 && weights_bytes) {
                std::snprintf(b, sizeof b, " workspace_mib=%d",
                              (int) std::max<int64_t>(0, gs.vram_mib - (int64_t) (weights_bytes >> 20)));
                tail += b;
            }
            std::snprintf(b, sizeof b, " chunks=%lld weights_mib=%d mel_ms=%.0f pre_ms=%.0f enc_ms=%.0f adapter_ms=%.0f",
                          (long long) enc.chunks, (int) (weights_bytes >> 20), enc.phases.mel_ms, enc.phases.pre_ms,
                          enc.phases.win_ms, enc.phases.adapter_ms);
            tail += b;
        }
        std::printf("OK %lld 0 %lld %.0f%s\n", (long long) n_rows, (long long) dur_ms, ms, tail.c_str());
        std::fflush(stdout);
    }
    return 0;
}
