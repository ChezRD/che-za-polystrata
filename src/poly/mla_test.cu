// src/poly/mla_test.cu - MLA's absorbed attention (kernels_mla.cu: mla_attend and the packing kernels) against
// the same arithmetic on the host.  Random latent and rope caches, a fused kv_b, no model: it checks that the
// rates, the causal mask and Q8_0's cache read are what the comments say.
//
// usage: poly_mla_test   (exit 0: every kernel within float rounding of the host)
#include "strata/poly/kernels.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

using namespace strata::poly;

std::mt19937 rng(11);

std::vector<float> random(size_t n, float scale = 1.f) {
    std::normal_distribution<float> nd(0.f, scale);
    std::vector<float> v(n);
    for (float& x : v) x = nd(rng);
    return v;
}

template <class T>
T* dev(const std::vector<T>& v) {
    T* p = nullptr;
    cudaMalloc(&p, std::max<size_t>(1, v.size()) * sizeof(T));
    cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return p;
}

template <class T>
std::vector<T> host(const T* p, size_t n) {
    std::vector<T> v(n);
    cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost);
    return v;
}

std::vector<uint16_t> to_f16(const std::vector<float>& v) {
    std::vector<uint16_t> h(v.size());
    for (size_t i = 0; i < v.size(); ++i) h[i] = __half_as_ushort(__float2half(v[i]));
    return h;
}

float f16(uint16_t h) { return __half2float(__ushort_as_half(h)); }

int report(const char* what, const std::vector<float>& got, const std::vector<float>& want, double tol) {
    double worst = 0.0;
    for (size_t i = 0; i < got.size(); ++i)
        worst = std::max(worst, std::abs((double) got[i] - want[i]) / (1e-3 + std::abs((double) want[i])));
    const bool ok = worst <= tol;
    std::printf("%-40s %zu values, worst %.2g: %s\n", what, got.size(), worst, ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}

// The host's attention: qabs/qpe are the query, lat/kpe the cache after whatever precision it was stored in.
std::vector<float> reference(const std::vector<float>& qabs, const std::vector<float>& qpe,
                             const std::vector<float>& lat, const std::vector<float>& kpe, int rank, int n_rot,
                             int n_head, int64_t pos, int rows, float scale) {
    std::vector<float> out((size_t) rows * n_head * rank);
    for (int t = 0; t < rows; ++t) {
        const int64_t p = pos + t;
        const int n_key = (int) p + 1;
        for (int h = 0; h < n_head; ++h) {
            std::vector<float> s((size_t) n_key);
            const float* qa = qabs.data() + ((size_t) t * n_head + h) * rank;
            const float* qr = qpe.data() + ((size_t) t * n_head + h) * n_rot;
            float mx = -INFINITY;
            for (int k = 0; k < n_key; ++k) {
                float d = 0;
                for (int i = 0; i < rank; ++i) d += qa[i] * lat[(size_t) k * rank + i];
                for (int i = 0; i < n_rot; ++i) d += qr[i] * kpe[(size_t) k * n_rot + i];
                s[(size_t) k] = d * scale;
                mx = std::max(mx, s[(size_t) k]);
            }
            float sum = 0;
            for (float& x : s) sum += x = std::exp(x - mx);
            for (int i = 0; i < rank; ++i) {
                float acc = 0;
                for (int k = 0; k < n_key; ++k) acc += s[(size_t) k] * lat[(size_t) k * rank + i];
                out[((size_t) t * n_head + h) * rank + i] = acc / sum;
            }
        }
    }
    return out;
}

// ggml Q8_0 as store_q8() writes it: one fp16 scale a 32-value block, a code = roundf(x / scale).
void quantize_q8(const std::vector<float>& x, int width, int rows, std::vector<int8_t>& code,
                 std::vector<uint16_t>& scale) {
    for (int r = 0; r < rows; ++r)
        for (int b = 0; b < width / 32; ++b) {
            const float* v = x.data() + (size_t) r * width + b * 32;
            float amax = 0;
            for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(v[j]));
            const float d = amax / 127.f, id = d != 0 ? 1.f / d : 0.f;
            scale[(size_t) r * (width / 32) + b] = __half_as_ushort(__float2half(d));
            for (int j = 0; j < 32; ++j) code[(size_t) r * width + b * 32 + j] = (int8_t) roundf(v[j] * id);
        }
}

// A Q8_0 cache as the kernel dequantizes it: one fp16 scale a 32-value block.
std::vector<float> deq(const std::vector<int8_t>& code, const std::vector<uint16_t>& scale, int width, int rows) {
    std::vector<float> v((size_t) width * rows);
    for (size_t i = 0; i < v.size(); ++i)
        v[i] = float(code[i]) * f16(scale[i / 32]);
    return v;
}

int test_attend(bool q8, bool use_at) {
    const int rank = 512, n_rot = 64, n_head = 32, pos = 5, rows = 3, n_tok = pos + rows;
    const float scale = 0.1447f;
    const auto latf = random((size_t) n_tok * rank, 0.6f), kpef = random((size_t) n_tok * n_rot, 0.6f);
    const auto qabs = random((size_t) rows * n_head * rank, 0.08f), qpe = random((size_t) rows * n_head * n_rot, 0.08f);

    std::vector<float> lat = latf, kpe = kpef;
    uint16_t *d_lat = nullptr, *d_kpe = nullptr;
    int8_t *d_lc = nullptr, *d_kc = nullptr;
    uint16_t *d_ls = nullptr, *d_ks = nullptr;
    Kvq8 lq, kq;
    if (q8) {
        std::vector<int8_t> lc((size_t) n_tok * rank), kc((size_t) n_tok * n_rot);
        std::vector<uint16_t> ls((size_t) n_tok * rank / 32), ks((size_t) n_tok * n_rot / 32);
        quantize_q8(latf, rank, n_tok, lc, ls);
        quantize_q8(kpef, n_rot, n_tok, kc, ks);
        d_lc = dev(lc);
        d_kc = dev(kc);
        d_ls = dev(ls);
        d_ks = dev(ks);
        lat = deq(lc, ls, rank, n_tok);
        kpe = deq(kc, ks, n_rot, n_tok);
        lq = Kvq8{d_lc, d_ls};
        kq = Kvq8{d_kc, d_ks};
    } else {
        d_lat = dev(to_f16(latf));
        d_kpe = dev(to_f16(kpef));
        for (float& x : lat) x = f16(__half_as_ushort(__float2half(x)));
        for (float& x : kpe) x = f16(__half_as_ushort(__float2half(x)));
    }

    float* d_qabs = dev(qabs);
    float* d_qpe = dev(qpe);
    const size_t qn = (size_t) rows * n_head * (rank + n_rot);
    float* d_q = dev(std::vector<float>(qn, 0.f));
    mla_pack_q(d_qabs, d_qpe, d_q, rank, n_rot, n_head, rows, nullptr);

    const int room = use_at ? pos + rows + 5 : 0;
    std::vector<long long> h_at((size_t) rows * 2);
    for (int t = 0; t < rows; ++t) {
        h_at[(size_t) 2 * t] = pos + t;
        h_at[(size_t) 2 * t + 1] = pos + t;
    }
    long long* d_at = use_at ? dev(h_at) : nullptr;

    const int chunk = 2;   // rows = 3: tests a pass of two and a pass of one
    const int n_keys = std::max(use_at ? room : pos + chunk, pos + rows);
    const size_t keyn = (size_t) chunk * n_keys * (rank + n_rot);
    float* d_keys = dev(std::vector<float>(keyn, 0.f));
    float* d_score = dev(std::vector<float>((size_t) chunk * n_keys * n_head, 0.f));
    const size_t outn = (size_t) rows * n_head * rank;
    float* d_out = dev(std::vector<float>(outn, 0.f));

    cublasHandle_t blas = nullptr;
    cublasCreate(&blas);
    cublasSetStream(blas, nullptr);
    mla_attend(blas, d_q, d_lat, d_kpe, pos, rows, d_keys, d_score, chunk, d_out, rank, n_rot, n_head, scale, nullptr,
               d_at, room, lq, kq);
    cudaDeviceSynchronize();
    const int err = cudaGetLastError() != cudaSuccess;
    cublasDestroy(blas);

    const auto got = host(d_out, outn);
    const auto want = reference(qabs, qpe, lat, kpe, rank, n_rot, n_head, pos, rows, scale);
    char what[96];
    std::snprintf(what, sizeof what, "mla_attend %s%s", q8 ? "q8_0" : "fp16", use_at ? ", recorded" : "");
    int bad = err ? 1 : report(what, got, want, q8 ? 4e-3 : 1e-3);
    if (err) std::printf("%-40s CUDA error\n", what);

    for (void* p : {(void*) d_lat, (void*) d_kpe, (void*) d_lc, (void*) d_kc, (void*) d_ls, (void*) d_ks,
                    (void*) d_qabs, (void*) d_qpe, (void*) d_q, (void*) d_at, (void*) d_keys, (void*) d_score,
                    (void*) d_out})
        cudaFree(p);
    return bad;
}

// The query and key splits: qn/qpe out of a 192-value head, and lat/kpe out of the 576 values kv_a gives.
int test_splits() {
    const int rank = 512, n_rot = 64, nope = 128, n_head = 32, rows = 2;
    const auto kv = random((size_t) rows * (rank + n_rot));
    const auto q = random((size_t) rows * n_head * (nope + n_rot));
    float* d_kv = dev(kv);
    float* d_q = dev(q);
    float* d_lat = dev(std::vector<float>((size_t) rows * rank, 0.f));
    float* d_kpe = dev(std::vector<float>((size_t) rows * n_rot, 0.f));
    float* d_qn = dev(std::vector<float>((size_t) rows * n_head * nope, 0.f));
    float* d_qpe = dev(std::vector<float>((size_t) rows * n_head * n_rot, 0.f));
    mla_split_kv(d_kv, d_lat, d_kpe, rank, n_rot, rows, nullptr);
    mla_split_q(d_q, d_qn, d_qpe, nope, n_rot, n_head, rows, nullptr);
    cudaDeviceSynchronize();
    std::vector<float> lat((size_t) rows * rank), kpe((size_t) rows * n_rot);
    std::vector<float> qn((size_t) rows * n_head * nope), qpe((size_t) rows * n_head * n_rot);
    for (int t = 0; t < rows; ++t) {
        for (int i = 0; i < rank; ++i) lat[(size_t) t * rank + i] = kv[(size_t) t * (rank + n_rot) + i];
        for (int i = 0; i < n_rot; ++i) kpe[(size_t) t * n_rot + i] = kv[(size_t) t * (rank + n_rot) + rank + i];
        for (int h = 0; h < n_head; ++h) {
            for (int i = 0; i < nope; ++i)
                qn[((size_t) t * n_head + h) * nope + i] = q[((size_t) t * n_head + h) * (nope + n_rot) + i];
            for (int i = 0; i < n_rot; ++i)
                qpe[((size_t) t * n_head + h) * n_rot + i] =
                    q[((size_t) t * n_head + h) * (nope + n_rot) + nope + i];
        }
    }
    int bad = report("mla_split_kv", host(d_lat, lat.size()), lat, 1e-6);
    bad += report("mla_split_kv rope", host(d_kpe, kpe.size()), kpe, 1e-6);
    bad += report("mla_split_q nope", host(d_qn, qn.size()), qn, 1e-6);
    bad += report("mla_split_q rope", host(d_qpe, qpe.size()), qpe, 1e-6);
    for (void* p : {(void*) d_kv, (void*) d_q, (void*) d_lat, (void*) d_kpe, (void*) d_qn, (void*) d_qpe})
        cudaFree(p);
    return bad;
}

// The legacy fused kv_b: the absorption and the value projection against the host's sums.
int test_fused(bool bf16) {
    const int nope = 128, rank = 512, vd = 192, n_head = 32, rows = 2, stride = nope + vd;
    const auto w = random((size_t) n_head * stride * rank, 0.05f);
    const auto qn = random((size_t) rows * n_head * nope, 0.3f);
    const auto kqv = random((size_t) rows * n_head * rank, 0.3f);

    std::vector<float> wq = w;
    std::vector<uint16_t> wb;
    if (bf16) {
        wb.resize(w.size());
        for (size_t i = 0; i < w.size(); ++i) {
            uint32_t bits = 0;
            std::memcpy(&bits, &w[i], sizeof(bits));
            wb[i] = (uint16_t) (bits >> 16);
            const uint32_t back = (bits >> 16) << 16;
            std::memcpy(&wq[i], &back, sizeof(back));
        }
    }
    void* d_w = bf16 ? (void*) dev(wb) : (void*) dev(w);
    float* d_qn = dev(qn);
    float* d_kqv = dev(kqv);
    float* d_qabs = dev(std::vector<float>((size_t) rows * n_head * rank, 0.f));
    float* d_vout = dev(std::vector<float>((size_t) rows * n_head * vd, 0.f));
    const uint32_t type = bf16 ? 30 : 0;
    mla_absorb_fused(d_qn, d_w, type, d_qabs, nope, rank, n_head, rows, stride, nullptr);
    mla_vb_fused(d_kqv, d_w, type, d_vout, rank, vd, n_head, rows, stride, nope, nullptr);
    cudaDeviceSynchronize();

    std::vector<float> wa((size_t) rows * n_head * rank), wv((size_t) rows * n_head * vd);
    for (int t = 0; t < rows; ++t)
        for (int h = 0; h < n_head; ++h) {
            const float* q = qn.data() + ((size_t) t * n_head + h) * nope;
            for (int l = 0; l < rank; ++l) {
                float acc = 0;
                for (int i = 0; i < nope; ++i) acc += q[i] * wq[((size_t) h * stride + i) * rank + l];
                wa[((size_t) t * n_head + h) * rank + l] = acc;
            }
            const float* kv = kqv.data() + ((size_t) t * n_head + h) * rank;
            for (int v = 0; v < vd; ++v) {
                float acc = 0;
                for (int l = 0; l < rank; ++l) acc += kv[l] * wq[((size_t) h * stride + nope + v) * rank + l];
                wv[((size_t) t * n_head + h) * vd + v] = acc;
            }
        }
    char what[64];
    std::snprintf(what, sizeof what, "fused kv_b %s, absorption", bf16 ? "bf16" : "f32");
    int bad = report(what, host(d_qabs, wa.size()), wa, 1e-4);
    std::snprintf(what, sizeof what, "fused kv_b %s, values", bf16 ? "bf16" : "f32");
    bad += report(what, host(d_vout, wv.size()), wv, 1e-4);
    for (void* p : {(void*) d_w, (void*) d_qn, (void*) d_kqv, (void*) d_qabs, (void*) d_vout}) cudaFree(p);
    return bad;
}

}  // namespace

int main() {
    int bad = test_attend(false, false);
    bad += test_attend(false, true);
    bad += test_attend(true, false);
    bad += test_attend(true, true);
    bad += test_splits();
    bad += test_fused(false);
    bad += test_fused(true);
    if (cudaGetLastError() != cudaSuccess) {
        std::printf("CUDA error\n");
        return 2;
    }
    return bad ? 1 : 0;
}
