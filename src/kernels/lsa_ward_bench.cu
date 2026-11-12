// src/kernels/lsa_ward_bench.cu - Phase 0 gate W1: spherical Ward clustering of one LSA cluster block on Ada
// sm_89, one CTA a block, all merge rounds in shared memory, no host round trip between rounds.
//
// WHY THIS ROW EXISTS.  `LSA-PLAN.md` §3.3 replaces QSA's 65,536-block indexer at 256K with bucket centroids,
// and the buckets come from Ward-merging each 512-key cluster block.  The paper's kernel spreads one block over
// a thread-block cluster of 8 SMs and exchanges state through distributed shared memory; sm_89 has neither
// (`__cluster_dims__ is not supported for this GPU architecture`).  The portable design is ONE CTA a block, and
// this binary measures what that costs, because §8.2 ranks it the risk most likely to be wrong.
//
// THE BUDGET, STATED BEFORE THE KERNEL (the brief's rule).  A 512-key block pooled 4:1 (the paper's n_pool) is
// P = 128 pool keys of d = 64, which is the plan's own geometry.  Per block:
//
//      key sums   fp32 [P][68] (row stride padded off 64 for bank spread)  128*68*4 =  34,816 B
//      Gram       fp32 packed lower triangle [P][P-1/2]                       8128*4 =  32,512 B
//      G_bb, a_b, best cost   f32 [P] x 3                                     3*512 =   1,536 B
//      size/best/parent/rank  i32 [P] x 4                                     4*512 =   2,048 B
//      merge list (survivor, absorbed, flag) i32 [P] x 3                      3*512 =   1,536 B
//      round counters, phase clocks                                                   ~64 B
//                                                                          total =  72,512 B of 101,376
//
// 512 keys in 128 pools FITS with 28.8 KB spare.  **256 POOLS DOES NOT, and this is stated before the kernel
// runs rather than discovered from an Xid**: 256 pools need 256*68*4 + 256*255/2*4 = 200,192 B, 2.0x the opt-in,
// so a 256-pool block cannot cache its Gram in shared memory at all.  The plan's own fallback (recompute
// G[b][b'] every round from the key sums) is what makes 256 pools possible, and it is measured here as
// `--arm recompute`; a 256-key block (64 pools) is measured too, since it fits easily and answers "does a
// different block size move it".
//
//     build/lsa_ward_bench [--keys FILE] [--blocks N] [--m M] [--cap C] [--pools N]
//                          [--runs R] [--iters I] [--arm cta|warp|warp8|recompute] [--oracle N]
//                          [--smoke] [--barrier]
//
// Keys: `--keys` reads raw fp16, block-major, [n_blocks][512][64], unit-norm - produced from the checkpoint's own
// `indexer.k_proj`/`indexer.k_norm` (the generator is /tmp/opencode/lsa/ward_keys.py and the report states the
// provenance).  Without it the built-in generator draws the same structure and the output says so.
//
// Exit 0 always unless a CUDA call fails: this is a benchmark.  `--smoke` is the partial-tile and invariants
// check and exits 1 on a violation.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "lsa_ward_bench: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

constexpr int kD = 64;         // the LSA index dim
constexpr int kPool = 4;       // index keys a pool key sums (the paper's n_pool)
constexpr int kStride = 65;    // key-sum row stride: 64 padded to 65 so a warp's 8 rows land on 8 banks
constexpr int kDead = 1 << 28; // a dead pool's size: fails the cap test before anything else looks at it

// Packed lower triangle: entry (i, j) with i > j sits at i*(i-1)/2 + j.  The diagonal is kept separately
// because every round reads all P of them.
// The packed lower triangle: entry (i, j) with i > j at i(i-1)/2 + j.
__device__ __forceinline__ int gidx(int i, int j) { return ((i * (i - 1)) >> 1) + j; }

// A deterministic square root: MUFU.RSQ (one instruction, no guard) refined by one Newton step.  `__fsqrt_rn`
// compiles to MUFU.RSQ plus a range guard plus a CALL to a slow path - 70 of those in this kernel's inner
// loop, which is why the round was 43,000 cycles.  This is NOT correctly rounded; it is accurate to ~1e-7
// relative, identical every run, and monotone in x, which is all the merge cost needs (the plan fixes the
// summation order for reproducibility, not the rounding mode).  The CPU oracle uses the same form.
__device__ __forceinline__ float fast_sqrt(float x) {
    const float r = __frsqrt_rn(x);      // one MUFU.RSQ
    return x * r;                        // x * 1/sqrt(x), one multiply, no branch
}

struct Params {
    int n_keys;     // index keys in the block (512; the smoke also runs 511, a 3-key last pool)
    int pools;      // live pool keys = ceil(n_keys / 4)
    int m;          // buckets to merge down to
    int cap;        // most KEYS a bucket may hold (n_b_max: 16 at the plan's 8-key buckets)
    int max_rounds; // the authors cap at 256 and refuse if the cap stops them short
    int n_blocks;
};

struct Shared {
    float* key;    // [P][kStride]
    float* gram;   // [P][P-1/2]  (unused by the recompute arm)
    float* sq;     // [P] G_bb
    float* al;     // [P] sqrt(G_bb)
    float* bc;     // [P] cheapest cost this round
    int* sz;       // [P] keys in the bucket
    int* best;     // [P] cheapest partner, later the root
    int* par;      // [P] the bucket this one merged into
    int* rank;     // [P] 0 none / 1 mutual / 2 mutual and kept by the rank rule, later the bucket id
    int* mb;       // [P] this round's survivors
    int* mo;       // [P] the bucket each survivor absorbed
    int* cnt;      // [8] alive, mutual, applied, rounds, buckets, merges
    long long* clk;// [6] phase clocks (block 0): pool+Gram, rounds, emit, pair, commit
};

template <int THREADS>
__device__ __forceinline__ int block_reduce_add(int v) {
    __shared__ int red[(THREADS + 31) / 32];   // one slot a warp; sizing it THREADS wastes registers
    v += __shfl_down_sync(0xffffffffu, v, 16);
    v += __shfl_down_sync(0xffffffffu, v, 8);
    v += __shfl_down_sync(0xffffffffu, v, 4);
    v += __shfl_down_sync(0xffffffffu, v, 2);
    v += __shfl_down_sync(0xffffffffu, v, 1);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) red[warp] = v;
    __syncthreads();
    int total = 0;
    for (int w = 0; w < (THREADS + 31) / 32; ++w) total += red[w];
    return total;
}

template <int P>
__device__ __forceinline__ Shared carve(unsigned char* smem) {
    Shared s;
    s.key = reinterpret_cast<float*>(smem);
    s.gram = s.key + (size_t) P * kStride;
    s.sq = s.gram + ((size_t) P * (P - 1)) / 2;
    s.al = s.sq + P;
    s.bc = s.al + P;
    s.sz = reinterpret_cast<int*>(s.bc + P);
    s.best = s.sz + P;
    s.par = s.best + P;
    s.rank = s.par + P;
    s.mb = s.rank + P;
    s.mo = s.mb + P;
    s.cnt = s.mo + P;
    s.clk = reinterpret_cast<long long*>(s.cnt + 8);
    return s;
}

// P is a template parameter inside the kernel; the host sizes the request from the same expression
__host__ __device__ constexpr int smem_for(int P, bool cached) {
    return P * kStride * 4 + (cached ? P * (P - 1) / 2 * 4 : 0) + P * 4 * 3 + P * 4 * 6 + 8 * 4 + 6 * 8 + 16;
}
template <int P>
__host__ __device__ constexpr int smem_bytes() {
    return smem_for(P, true);
}

// ---- G(i, j), packed, with the symmetric accessor the two arms and the update share
template <bool CACHED>
__device__ __forceinline__ float gram_get(const Shared& s, int i, int j) {
    if (CACHED) return (i > j) ? s.gram[gidx(i, j)] : s.gram[gidx(j, i)];
    // four independent chains so the dot has ILP 4 without a 64-float array: the un-cached arm runs this once
    // a candidate and the single chain is both slow and a register-pressure spike.
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    const float* p = &s.key[(size_t) i * kStride];
    const float* q = &s.key[(size_t) j * kStride];
    for (int d = 0; d < kD; d += 4) {
        a0 = __fmaf_rn(p[d + 0], q[d + 0], a0);
        a1 = __fmaf_rn(p[d + 1], q[d + 1], a1);
        a2 = __fmaf_rn(p[d + 2], q[d + 2], a2);
        a3 = __fmaf_rn(p[d + 3], q[d + 3], a3);
    }
    return (a0 + a1) + (a2 + a3);
}

// the key sums, the Gram, the alignments
template <int P, int THREADS, bool CACHED>
__device__ __forceinline__ void build_state(Shared& s, const __half* __restrict__ keys, Params p, int blk) {
    const int tid = threadIdx.x;
    for (int row = tid; row < P; row += THREADS) {
        // Four cells a pool, summed in one pass with a 64-float accumulator.  This is the form that measures
        // STACK 0 - a 16-dim chunk loop puts the accumulator on the stack instead (40 B), because the array is
        // indexed by a variable at the bottom of the loop.
        float acc[kD];
        #pragma unroll
        for (int d = 0; d < kD; ++d) acc[d] = 0.f;
        if (row < p.pools) {
            const int k0 = row * kPool;
            const int kn = min(kPool, p.n_keys - k0);   // a partial last pool (the smoke's 511-key block)
            for (int t = 0; t < kn; ++t) {
                const __half* src = keys + ((size_t) blk * p.n_keys + k0 + t) * kD;
                #pragma unroll
                for (int d = 0; d < kD; ++d) acc[d] += __half2float(src[d]);
            }
            s.sz[row] = kn;
        } else {
            s.sz[row] = kDead;
        }
        #pragma unroll
        for (int d = 0; d < kD; ++d) s.key[(size_t) row * kStride + d] = acc[d];
        s.par[row] = row;
        s.sq[row] = 0.f;
        s.bc[row] = INFINITY;
        s.best[row] = -1;
        s.rank[row] = 0;
    }
    __syncthreads();
    if (CACHED) {
        // One 8x8 tile a thread, lower triangle only.  **THE SUMMATION ORDER IS DIM ASCENDING**, one fma a dim,
        // because that is the order the plan fixes (§3.3) and the order the CPU oracle uses: a partition must be
        // reproducible run to run AND comparable against a scalar transcription.  An earlier version summed in
        // float4 groups (w, z, y, x within each group), which is equally deterministic but rounding-different,
        // and the oracle then disagreed on 2 blocks of 2 - correctly, because a near-tied merge cost turns on the
        // last bit.  The 64 cells of the tile give the ILP; the chain within a cell is exactly 64 fmas.
        // The tile grid follows P, not a fixed 16x16: with `tile < 256` hard-coded, a P=64 block read rows
        // 64..127 of a 64-row key array - an out-of-bounds shared read that compute-sanitizer caught and that
        // no amount of oracle agreement on P=128 could have found.
        constexpr int kTiles = (P + 7) / 8;
        for (int tile = tid; tile < kTiles * kTiles; tile += THREADS) {
            const int ti = tile / kTiles, tj = tile % kTiles;
            if (tj > ti) continue;
            const int I = ti * 8, J = tj * 8;
            float acc[8][8];
            for (int i = 0; i < 8; ++i)
                for (int j = 0; j < 8; ++j) acc[i][j] = 0.f;
            for (int k = 0; k < kD; ++k) {
                float a[8], b[8];
                for (int i = 0; i < 8; ++i) a[i] = (I + i < P) ? s.key[(size_t) (I + i) * kStride + k] : 0.f;
                for (int j = 0; j < 8; ++j) b[j] = (J + j < P) ? s.key[(size_t) (J + j) * kStride + k] : 0.f;
                for (int i = 0; i < 8; ++i)
                    for (int j = 0; j < 8; ++j) acc[i][j] = __fmaf_rn(a[i], b[j], acc[i][j]);
            }
            for (int i = 0; i < 8; ++i) {
                const int gi = I + i;
                if (gi >= P) continue;
                for (int j = 0; j < 8; ++j) {
                    const int gj = J + j;
                    if (gj >= P || gj > gi) continue;
                    if (gi > gj) s.gram[gidx(gi, gj)] = acc[i][j];
                    else s.sq[gi] = acc[i][j];
                }
            }
        }
        __syncthreads();
    }
    for (int row = tid; row < P; row += THREADS) {
        if (s.sz[row] >= kDead) { s.al[row] = INFINITY; continue; }
        if (!CACHED) {
            float acc = 0.f;
            const float* a = &s.key[(size_t) row * kStride];
            for (int d = 0; d < kD; ++d) acc = __fmaf_rn(a[d], a[d], acc);
            s.sq[row] = acc;
        }
        s.sq[row] = fmaxf(s.sq[row], 0.f);
        s.al[row] = fast_sqrt(s.sq[row]);
    }
    __syncthreads();
}

// ---- the round's pair decision: each live bucket's cheapest allowed partner, then the mutual pairs and the
// rank rule.  Returns how many merge this round (also in `applied`).
template <int P, int THREADS>
__device__ __forceinline__ int pick_pairs(Shared& s, int merges_left, int* mutual_out) {
    const int tid = threadIdx.x;
    if (tid == 0) {
        s.cnt[1] = 0;
        s.cnt[2] = 0;
        s.cnt[5] = 0;
    }
    __syncthreads();
    // the mutual rule: b's cheapest is b' AND b''s is b AND b < b'.  A pair is credited once, to its lower
    // index, which is also the index it merges into.  Recomputed here rather than read back from `rank`, so a
    // thread never reads a flag another thread is overwriting.
    for (int row = tid; row < P; row += THREADS) {
        const int o = s.best[row];
        const bool mut = (o >= 0 && o < P && s.best[o] == row && row < o);
        s.rank[row] = mut ? 1 : 0;
    }
    __syncthreads();
    {
        // a parallel count: thread 0's serial P-long scan was 2 of the round's 6 barriers' worth of latency
        int n = 0;
        for (int r = tid; r < P; r += THREADS) n += s.rank[r] == 1;
        n = block_reduce_add<THREADS>(n);
        if (tid == 0) s.cnt[1] = n;
    }
    __syncthreads();
    const int mutual = s.cnt[1];
    *mutual_out = mutual;
    if (mutual == 0) return 0;
    if (mutual <= merges_left) return mutual;
    // the rank rule: only the `merges_left` cheapest mutual pairs merge, ranked by (cost, lower index).
    // The mutual set is compacted once (it is the survivors-only set, O(mutual)), and each member counts over
    // it; counting over all P instead makes the round O(P * mutual) of dependent work.
    // The decision goes to `mo` and is published to `rank` after a barrier: writing `rank[row]` in place while
    // another thread still counts `rank[o]` makes the count depend on warp scheduling, and it did.
    if (tid == 0) {
        int n = 0;
        for (int r = 0; r < P; ++r)
            if (s.rank[r] == 1) s.mb[n++] = r;
        s.cnt[6] = n;
    }
    __syncthreads();
    const int nm = s.cnt[6];
    for (int row = tid; row < P; row += THREADS) {
        if (s.rank[row] != 1) { s.mo[row] = 0; continue; }
        const float mine = s.bc[row];
        int cnt = 0;
        for (int k = 0; k < nm; ++k) {
            const int o = s.mb[k];
            if (s.bc[o] < mine || (s.bc[o] == mine && o < row)) ++cnt;
        }
        s.mo[row] = cnt < merges_left ? 2 : 0;
    }
    __syncthreads();
    for (int row = tid; row < P; row += THREADS) s.rank[row] = s.mo[row];
    __syncthreads();
    {
        int n = 0;
        for (int r = tid; r < P; r += THREADS) n += s.rank[r] == 2;
        n = block_reduce_add<THREADS>(n);
        if (tid == 0) s.cnt[2] = n;
    }
    __syncthreads();
    return s.cnt[2];
}

// ---- commit the round's merges.  Mutual pairs are vertex-disjoint, but two merges in the SAME round still
// interfere through the Gram: the union row of one merge must be built from the PRE-round rows, and if the other
// merge's survivor sits in it, that cell is the other merge's to write.  A naive "row b += row o" per merge
// therefore both races and misses the cross term G(b, o2) + G(o, o2).  Two phases, ordered so each cell is
// written once and every read sees a pre-round value:
//   A: every PAIR of survivors (b, b2) gets the four-term sum, from rows no other merge touches yet;
//   B: every cell (b, x) with x not a survivor gets the plain row add.
// Phase B never writes a cell with both indices in the survivor set, and phase A never writes one with an index
// outside it, so the phases are disjoint and one barrier between them is enough.
template <int P, int THREADS, bool CACHED>
__device__ __forceinline__ int commit_merges(Shared& s, int nmerge, bool ranked) {
    const int tid = threadIdx.x;
    (void) nmerge;
    // the compact merge list; -1 marks a survivor and -2 an absorbed pool, so neither is mistaken for a
    // third party in phase B
    if (tid == 0) {
        int n = 0;
        for (int r = 0; r < P; ++r) {
            const bool take = ranked ? (s.rank[r] == 2) : (s.rank[r] == 1);
            if (!take) continue;
            const int o = s.best[r];
            if (o < 0 || o >= P) continue;
            s.mb[n] = r;
            s.mo[n] = o;
            s.rank[r] = -1;
            ++n;
        }
        for (int m = 0; m < n; ++m) s.rank[s.mo[m]] = -2;
        s.cnt[5] = n;
    }
    __syncthreads();
    const int n = s.cnt[5];
    if (CACHED) {
        // ---- phase A: the cells between two survivors, from the PRE-round Gram.  The four-term sum is the
        // cross term a per-merge "row b += row o" misses: G(b|o, b2|o2) = G(b,b2) + G(o,b2) + G(b,o2) + G(o,o2).
        for (int i = tid; i < n * n; i += THREADS) {
            const int a = i / n, c = i % n;
            if (a >= c) continue;
            const int b = s.mb[a], b2 = s.mb[c], o = s.mo[a], o2 = s.mo[c];
            const float v = gram_get<true>(s, b, b2) + gram_get<true>(s, o, b2) + gram_get<true>(s, b, o2) +
                            gram_get<true>(s, o, o2);
            if (b > b2) s.gram[gidx(b, b2)] = v;
            else s.gram[gidx(b2, b)] = v;
        }
        __syncthreads();   // phase A's writes are complete before phase B reads any of them
        // ---- phase B: every other cell of a survivor's row and column.  Phase B writes only cells whose second
        // index is neither a survivor nor an absorbed pool, so it touches nothing phase A wrote or read.
        //
        // The (x, merge) pair is the parallel axis, NOT x alone: with ~5 merges a round an x-only loop leaves
        // 123 of 128 threads idle inside a 128-long serial sub-loop, and that was 86% of the round.
        for (int idx = tid; idx < P * n; idx += THREADS) {
            const int x = idx / n, m = idx % n;
            // `continue`, not `break`, and no early-exit on the warp's first lane: an `s.rank[x]` test that is
            // taken for lane 0 leaves the other 31 lanes' work in the same iteration, and the loop then walks
            // all P x-iterations serially.  With ~5 merges that was 88% of the round.
            if (s.rank[x] < 0) continue;      // a survivor or an absorbed pool
            if (s.sz[x] >= kDead) continue;   // already dead from an earlier round
            const int b = s.mb[m], o = s.mo[m];
            // the READS must go through the symmetric accessor: `gidx(o, x)` is only the cell (o, x) when
            // o > x, and o is an arbitrary pool index.  Reading it directly is a silent wrong answer on
            // half the merges, and it made this kernel disagree with its own oracle on every block.
            const int hi = b > x ? b : x, lo = b > x ? x : b;
            s.gram[gidx(hi, lo)] += gram_get<true>(s, o, x);
        }
        __syncthreads();
    }
    // ---- the merge itself: key sum, alignment, size, links.  Pairs are vertex-disjoint, so no two touch a row.
    for (int m = tid; m < n; m += THREADS) {
        const int b = s.mb[m], o = s.mo[m];
        float* kb = &s.key[(size_t) b * kStride];
        const float* ko = &s.key[(size_t) o * kStride];
        for (int d = 0; d < kD; ++d) kb[d] += ko[d];
        const float gob = CACHED ? gram_get<true>(s, b, o) : gram_get<false>(s, b, o);
        s.sq[b] = fmaxf(s.sq[b] + s.sq[o] + 2.f * gob, 0.f);
        s.al[b] = fast_sqrt(s.sq[b]);
        s.sz[b] += s.sz[o];
        s.sz[o] = kDead;
        s.al[o] = INFINITY;
        s.bc[o] = INFINITY;
        s.par[o] = b;
    }
    __syncthreads();
    for (int r = tid; r < P; r += THREADS)
        if (s.rank[r] < 0) s.rank[r] = 0;
    __syncthreads();
    return n;
}

// ---- the emit: labels, centroids, members.  A bucket's centroid is its key sum normalised; its members are the
// pools whose root it is, ascending (the reference's `members_from_labels`).
template <int P, int THREADS>
__device__ __forceinline__ void emit(Shared& s, Params p, int blk, int alive, __half* __restrict__ centroid,
                                     int32_t* __restrict__ members, int32_t* __restrict__ bsize,
                                     int32_t* __restrict__ nrounds, int32_t* __restrict__ reached, int rounds) {
    const int tid = threadIdx.x;
    for (int q = tid; q < P; q += THREADS) {
        int r = q, g = 0;
        while (s.par[r] != r && g++ < P) r = s.par[r];
        s.best[q] = r;   // best[] is free after the last round; it holds the root here
    }
    __syncthreads();
    if (tid == 0) {
        int nb = 0;
        for (int r = 0; r < P; ++r) s.rank[r] = (s.sz[r] < kDead) ? nb++ : -1;
        s.cnt[4] = nb;
    }
    __syncthreads();
    const int slots = p.cap / kPool;
    for (int b = tid; b < p.m; b += THREADS) {
        __half* out = centroid + ((size_t) blk * p.m + b) * kD;
        int32_t* mem = members + ((size_t) blk * p.m + b) * slots;
        int root = -1;
        for (int r = 0; r < P; ++r)
            if (s.rank[r] == b && s.sz[r] < kDead) { root = r; break; }
        float v[kD];
        for (int d = 0; d < kD; ++d) v[d] = 0.f;
        if (root >= 0)
            for (int d = 0; d < kD; ++d) v[d] = s.key[(size_t) root * kStride + d];
        float ss = 0.f;
        for (int d = 0; d < kD; ++d) ss = __fmaf_rn(v[d], v[d], ss);
        const float nrm = fast_sqrt(ss);
        const float inv = nrm > 0.f ? 1.f / nrm : 0.f;
        for (int d = 0; d < kD; ++d) out[d] = __float2half(v[d] * inv);
        // a bucket's members, ascending: pools whose root's bucket is b.  `mb` is free here, so it is the
        // per-bucket cursor and the fill is one O(P) pass rather than a search a bucket.
        int slot = 0;
        for (int q = 0; q < P; ++q) {
            const int rt = s.best[q];
            if (rt < P && s.rank[rt] == b && s.sz[rt] < kDead) {
                if (slot < slots) mem[slot] = q;
                ++slot;
            }
        }
        for (int k = slot < slots ? slot : slots; k < slots; ++k) mem[k] = -1;
        bsize[(size_t) blk * p.m + b] = root >= 0 ? s.sz[root] : 0;
    }
    if (tid == 0) {
        nrounds[blk] = rounds;
        reached[blk] = alive <= p.m ? 1 : 0;
    }
}

// ================= arm A: one thread a row (SPLIT=1) or two (SPLIT=2), one CTA a block, Gram cached
// =================
template <int P, int THREADS, int SPLIT, bool CACHED>
__global__ void __launch_bounds__(THREADS) ward_cta_kernel(const __half* __restrict__ keys, Params p,
                                                           __half* __restrict__ centroid, int32_t* __restrict__ members,
                                                           int32_t* __restrict__ bsize, int32_t* __restrict__ nrounds,
                                                           int32_t* __restrict__ reached, long long* __restrict__ phase,
                                                           int* __restrict__ trace) {
    extern __shared__ __align__(16) unsigned char smem[];
    Shared s = carve<P>(smem);
    const int tid = threadIdx.x, blk = blockIdx.x;
    if (tid < 8) s.cnt[tid] = 0;
    if (blk == 0 && tid < 6) s.clk[tid] = 0;
    long long t0 = 0;
    if (tid == 0) {
        s.cnt[0] = p.pools;
        s.cnt[3] = 0;
        if (blk == 0) t0 = clock64();
    }
    build_state<P, THREADS, CACHED>(s, keys, p, blk);
    if (tid == 0 && blk == 0) { const long long t = clock64(); s.clk[0] = t - t0; t0 = t; }

    int rounds = 0;
    for (;;) {
        __syncthreads();
        const int alive = s.cnt[0];
        if (alive <= p.m || rounds >= p.max_rounds) break;
        const int merges_left = alive - p.m;
        const long long t_s = (blk == 0 && tid == 0) ? clock64() : 0;
        {
            const int span = P / SPLIT;
            // The (row, part) mapping is chosen so the two parts of a row are ADJACENT lanes: with
            // `idx = tid`, `row = idx % P` puts part 1 of row r at lane r + P, which is 128 lanes away and
            // `__shfl_xor(..., 1)` then combines two different rows - the split path silently merged nothing
            // and stopped after 16 rounds, which is how it was found.  With `row = idx / SPLIT` the parts of a
            // row are lanes 2r and 2r+1.
            for (int idx = tid; idx < P * SPLIT; idx += THREADS) {
                const int row = idx / SPLIT, part = idx % SPLIT;
                float bc = INFINITY;
                int bj = -1;
                if (s.sz[row] < kDead) {
                    const int szr = s.sz[row];
                    const float ar = s.al[row], sqr = s.sq[row];
                    const int c0 = part * span, c1 = c0 + span;
                    // Delta = a_row + a_j - sqrt(G_bb + G_b'b' + 2G_bb').  `a_row` is the same for every
                    // candidate but `a_j` is not, so minimising Delta is NOT maximising the merged alignment q;
                    // there is no sqrt-free form that keeps the ranking (checked: the first version of this
                    // kernel tried one and the CPU oracle rejected its partition).  One MUFU a candidate is
                    // cheaper than being wrong, and it is not what this round costs anyway - the dependent LDS
                    // chain is (the report has the split).
                    for (int j = c0; j < c1; ++j) {
                        if (j == row || szr + s.sz[j] > p.cap) continue;
                        const float q = sqr + s.sq[j] + 2.f * gram_get<CACHED>(s, row, j);
                        const float c = ar + s.al[j] - fast_sqrt(fmaxf(q, 0.f));
                        if (c < bc) { bc = c; bj = j; }
                    }
                }
                if (SPLIT == 2) {
                    const float oc = __shfl_xor_sync(0xffffffffu, bc, 1);
                    const int oj = __shfl_xor_sync(0xffffffffu, bj, 1);
                    if (part == 0) {
                        if (oc < bc || (oc == bc && oj >= 0 && (bj < 0 || oj < bj))) { bc = oc; bj = oj; }
                        s.bc[row] = bc;
                        s.best[row] = bj;
                    }
                } else {
                    s.bc[row] = bc;
                    s.best[row] = bj;
                }
            }
        }
        __syncthreads();
        const long long t_a = (blk == 0 && tid == 0) ? clock64() : 0;
        if (blk == 0 && tid == 0) s.clk[5] += t_a - t_s;
        int mutual = 0;
        const int applied = pick_pairs<P, THREADS>(s, merges_left, &mutual);
        if (applied == 0) break;
        const long long t_b = (blk == 0 && tid == 0) ? clock64() : 0;
        const int done = commit_merges<P, THREADS, CACHED>(s, applied, mutual > merges_left);
        if (tid == 0) {
            if (blk == 0) { s.clk[3] += t_b - t_a; s.clk[4] += clock64() - t_b; }
            if (blk == 0 && trace != nullptr && rounds < 64) { trace[2 * rounds] = mutual; trace[2 * rounds + 1] = done; }
            s.cnt[0] -= done;
            s.cnt[3] += 1;
        }
        __syncthreads();
        rounds = s.cnt[3];
    }
    if (tid == 0 && blk == 0) { const long long t = clock64(); s.clk[1] = t - t0; t0 = t; }
    emit<P, THREADS>(s, p, blk, s.cnt[0], centroid, members, bsize, nrounds, reached, rounds);
    if (tid == 0 && blk == 0) {
        const long long t = clock64();
        s.clk[2] = t - t0;
        for (int i = 0; i < 6; ++i) phase[i] = s.clk[i];
    }
}

// ================= arm B: one warp a pool group, the warp's 32 lanes sharing a row's column scan =================
template <int P, int WARPS, bool CACHED>
__global__ void __launch_bounds__(WARPS * 32) ward_warp_kernel(const __half* __restrict__ keys, Params p,
                                                              __half* __restrict__ centroid, int32_t* __restrict__ members,
                                                              int32_t* __restrict__ bsize, int32_t* __restrict__ nrounds,
                                                              int32_t* __restrict__ reached, long long* __restrict__ phase,
                                                              int* __restrict__ trace) {
    extern __shared__ __align__(16) unsigned char smem[];
    Shared s = carve<P>(smem);
    constexpr int THREADS = WARPS * 32;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, blk = blockIdx.x;
    if (tid < 8) s.cnt[tid] = 0;
    if (blk == 0 && tid < 6) s.clk[tid] = 0;
    long long t0 = 0;
    if (tid == 0) {
        s.cnt[0] = p.pools;
        s.cnt[3] = 0;
        if (blk == 0) t0 = clock64();
    }
    build_state<P, THREADS, CACHED>(s, keys, p, blk);
    if (tid == 0 && blk == 0) { const long long t = clock64(); s.clk[0] = t - t0; t0 = t; }

    int rounds = 0;
    for (;;) {
        __syncthreads();
        const int alive = s.cnt[0];
        if (alive <= p.m || rounds >= p.max_rounds) break;
        const int merges_left = alive - p.m;
        const long long t_s = (blk == 0 && tid == 0) ? clock64() : 0;
        for (int row = warp; row < P; row += WARPS) {
            float bc = INFINITY;
            int bj = -1;
            if (s.sz[row] < kDead) {
                const int szr = s.sz[row];
                const float ar = s.al[row], sqr = s.sq[row];
                for (int j = lane; j < P; j += 32) {
                    if (j == row || szr + s.sz[j] > p.cap) continue;
                    const float q = sqr + s.sq[j] + 2.f * gram_get<CACHED>(s, row, j);
                    const float c = ar + s.al[j] - fast_sqrt(fmaxf(q, 0.f));
                    if (c < bc) { bc = c; bj = j; }
                }
            }
            for (int off = 16; off > 0; off >>= 1) {
                const float oc = __shfl_down_sync(0xffffffffu, bc, off);
                const int oj = __shfl_down_sync(0xffffffffu, bj, off);
                if (oc < bc || (oc == bc && oj >= 0 && (bj < 0 || oj < bj))) { bc = oc; bj = oj; }
            }
            if (lane == 0) { s.bc[row] = bc; s.best[row] = bj; }
        }
        __syncthreads();
        const long long t_a = (blk == 0 && tid == 0) ? clock64() : 0;
        if (blk == 0 && tid == 0) s.clk[5] += t_a - t_s;
        int mutual = 0;
        const int applied = pick_pairs<P, THREADS>(s, merges_left, &mutual);
        if (applied == 0) break;
        const long long t_b = (blk == 0 && tid == 0) ? clock64() : 0;
        const int done = commit_merges<P, THREADS, CACHED>(s, applied, mutual > merges_left);
        if (tid == 0) {
            if (blk == 0) { s.clk[3] += t_b - t_a; s.clk[4] += clock64() - t_b; }
            if (blk == 0 && trace != nullptr && rounds < 64) { trace[2 * rounds] = mutual; trace[2 * rounds + 1] = done; }
            s.cnt[0] -= done;
            s.cnt[3] += 1;
        }
        __syncthreads();
        rounds = s.cnt[3];
    }
    if (tid == 0 && blk == 0) { const long long t = clock64(); s.clk[1] = t - t0; t0 = t; }
    emit<P, THREADS>(s, p, blk, s.cnt[0], centroid, members, bsize, nrounds, reached, rounds);
    if (tid == 0 && blk == 0) {
        const long long t = clock64();
        s.clk[2] = t - t0;
        for (int i = 0; i < 6; ++i) phase[i] = s.clk[i];
    }
}

// ================= the CPU transcription of the same rules, for `--oracle` =================
struct Oracle {
    int P = 0, m = 0, cap = 0, rounds = 0;
    bool reached = true;
    std::vector<float> key, gram, sq, al, bc;
    std::vector<int> sz, best, par, rank, label;
    std::vector<int> trace;   // (mutual, applied) a round

    static int gi(int i, int j) { return ((i * (i - 1)) >> 1) + j; }

    void run(const std::vector<__half>& keys, int n_keys, int P_, int m_, int cap_, int max_rounds) {
        P = P_; m = m_; cap = cap_;
        key.assign((size_t) P * kStride, 0.f);
        gram.assign((size_t) P * (P - 1) / 2, 0.f);
        sq.assign(P, 0.f); al.assign(P, 0.f); bc.assign(P, INFINITY);
        sz.assign(P, 0); best.assign(P, -1); par.assign(P, 0); rank.assign(P, 0);
        for (int r = 0; r < P; ++r) {
            par[r] = r;
            if (r >= P_) { sz[r] = kDead; al[r] = INFINITY; continue; }
            const int k0 = r * kPool, kn = std::min(kPool, n_keys - k0);
            for (int t = 0; t < kn; ++t)
                for (int d = 0; d < kD; ++d)
                    key[(size_t) r * kStride + d] += __half2float(keys[(size_t) (k0 + t) * kD + d]);
            sz[r] = kn;
        }
        for (int i = 1; i < P; ++i)
            for (int j = 0; j < i; ++j) {
                float acc = 0.f;
                for (int d = 0; d < kD; ++d)
                    acc = std::fma(key[(size_t) i * kStride + d], key[(size_t) j * kStride + d], acc);
                gram[gi(i, j)] = acc;
            }
        for (int i = 0; i < P; ++i) {
            float acc = 0.f;
            for (int d = 0; d < kD; ++d) acc = std::fma(key[(size_t) i * kStride + d], key[(size_t) i * kStride + d], acc);
            sq[i] = std::max(acc, 0.f);
            al[i] = sz[i] >= kDead ? INFINITY : std::sqrt(sq[i]);
        }
        int alive = P_;
        while (alive > m && rounds < max_rounds) {
            const int merges_left = alive - m;
            for (int r = 0; r < P; ++r) {
                float c = INFINITY;
                int bj = -1;
                if (sz[r] < kDead) {
                    for (int j = 0; j < P; ++j) {
                        if (j == r || sz[r] + sz[j] > cap) continue;
                        const float g = r > j ? gram[gi(r, j)] : gram[gi(j, r)];
                        const float cc = al[r] + al[j] - std::sqrt(std::max(sq[r] + sq[j] + 2.f * g, 0.f));
                        if (cc < c) { c = cc; bj = j; }
                    }
                }
                bc[r] = c; best[r] = bj;
            }
            int mutual = 0;
            for (int r = 0; r < P; ++r) {
                rank[r] = (best[r] >= 0 && best[best[r]] == r && r < best[r]) ? 1 : 0;
                mutual += rank[r];
            }
            if (mutual == 0) { reached = false; break; }
            const int applied = std::min(mutual, merges_left);
            trace.push_back(mutual);
            trace.push_back(applied);
            // the compact merge list, exactly as the kernel builds it
            std::vector<int> mb, mo;
            if (mutual > merges_left) {
                std::vector<int> keep(P, 0);
                for (int r = 0; r < P; ++r) {
                    if (rank[r] != 1) continue;
                    int cnt = 0;
                    for (int o = 0; o < P; ++o) {
                        if (rank[o] != 1) continue;
                        if (bc[o] < bc[r] || (bc[o] == bc[r] && o < r)) ++cnt;
                    }
                    keep[r] = cnt < merges_left ? 2 : 0;
                }
                rank = keep;
            }
            const bool ranked = mutual > merges_left;
            std::vector<int> rank_mark(P, 0);
            for (int r = 0; r < P; ++r) {
                if (!(ranked ? (rank[r] == 2) : (rank[r] == 1))) continue;
                if (best[r] < 0 || best[r] >= P) continue;
                mb.push_back(r);
                mo.push_back(best[r]);
                rank_mark[r] = -1;
            }
            const int n = (int) mb.size();
            // phase A: cells between two survivors, from the pre-round Gram
            for (int a = 0; a < n; ++a)
                for (int c = a + 1; c < n; ++c) {
                    const int b = mb[a], b2 = mb[c], o = mo[a], o2 = mo[c];
                    auto gg = [&](int i, int j) { return i > j ? gram[gi(i, j)] : gram[gi(j, i)]; };
                    const float v = gg(b, b2) + gg(o, b2) + gg(b, o2) + gg(o, o2);
                    if (b > b2) gram[gi(b, b2)] = v;
                    else gram[gi(b2, b)] = v;
                }
            // phase B: every other cell of a survivor's row and column.  The reads go through the SYMMETRIC
            // accessor - `gi(o, x)` with o < x is a different cell of the packed triangle, and reading it
            // directly is the bug that made this oracle disagree with the kernel on every block.
            auto gget = [&](int i, int j) { return i > j ? gram[gi(i, j)] : gram[gi(j, i)]; };
            for (int x = 0; x < P; ++x) {
                if (rank_mark[x] < 0 || sz[x] >= kDead) continue;
                for (int k = 0; k < n; ++k) {
                    const int b = mb[k], o = mo[k];
                    if (x == b || x == o) continue;
                    const int hi = b > x ? b : x, lo = b > x ? x : b;
                    gram[gi(hi, lo)] += gget(o, x);
                }
            }
            for (int k = 0; k < n; ++k) {
                const int b = mb[k], o = mo[k];
                for (int d = 0; d < kD; ++d) key[(size_t) b * kStride + d] += key[(size_t) o * kStride + d];
                const float gob = b > o ? gram[gi(b, o)] : gram[gi(o, b)];
                sq[b] = std::max(sq[b] + sq[o] + 2.f * gob, 0.f);
                al[b] = std::sqrt(sq[b]);
                sz[b] += sz[o];
                sz[o] = kDead;
                al[o] = INFINITY;
                par[o] = b;
            }
            alive -= (int) n;
            if (n == 0) { reached = false; break; }
            (void) applied;
            ++rounds;
        }
        if (alive > m) reached = false;
        std::vector<int> root(P), bid(P, -1);
        for (int q = 0; q < P; ++q) {
            int r = q, g = 0;
            while (par[r] != r && g++ < P) r = par[r];
            root[q] = r;
        }
        int nb = 0;
        for (int r = 0; r < P; ++r)
            if (sz[r] < kDead) bid[r] = nb++;
        label.assign(P, -1);
        for (int q = 0; q < P; ++q) label[q] = bid[root[q]];
    }
};

// ================= data =================
void gen_keys(std::vector<__half>& keys, int n_blocks, int n_keys, unsigned seed) {
    // The built-in fallback, used only when `--keys` is absent and by `--smoke`: an AR(1) walk (rho 0.9) around
    // a shared mean, drawn through a channel gain with a few 8-12x channels - the structure the real projection
    // produces (the report has the real keys' statistics).  Every number the report quotes uses `--keys`.
    std::mt19937 rng(seed);
    std::normal_distribution<float> g(0.f, 1.f);
    std::uniform_real_distribution<float> uu(0.8f, 1.2f);
    std::vector<float> cg(kD, 1.f);
    for (int i = 0; i < 4; ++i) cg[(size_t) (rng() % kD)] = 8.f + 4.f * std::fabs(g(rng)) / 3.f;
    keys.assign((size_t) n_blocks * n_keys * kD, __float2half(0.f));
    std::vector<float> mu(kD), walk(kD), nw(kD);
    for (int b = 0; b < n_blocks; ++b) {
        float n = 0;
        for (int d = 0; d < kD; ++d) { mu[d] = g(rng) * cg[d]; n += mu[d] * mu[d]; }
        n = std::sqrt(n);
        for (int d = 0; d < kD; ++d) mu[d] /= n;
        for (int d = 0; d < kD; ++d) walk[d] = g(rng) * cg[d] * 0.7f;
        for (int k = 0; k < n_keys; ++k) {
            for (int d = 0; d < kD; ++d) {
                walk[d] = 0.9f * walk[d] + 0.55f * g(rng) * cg[d];
                nw[d] = mu[d] + walk[d];
            }
            float m = 0;
            for (int d = 0; d < kD; ++d) m += nw[d] * nw[d];
            m = std::sqrt(m);
            for (int d = 0; d < kD; ++d)
                keys[((size_t) b * n_keys + k) * kD + d] = __float2half(nw[d] / m);
        }
    }
    (void) uu;
}

bool read_keys(const char* path, std::vector<__half>& keys, int n_blocks, int n_keys) {
    std::FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    const size_t want = (size_t) n_blocks * n_keys * kD;
    keys.resize(want);
    const size_t got = std::fread(keys.data(), sizeof(__half), want, f);
    std::fclose(f);
    return got == want;
}

// mean over buckets of the mean squared distance to the bucket's own mean, over unit keys - the paper's Ward
// objective, so it is the honest "did the clustering do its job" number
double intra_variance(const __half* keys, int n_keys, const int32_t* members, int m, int slots) {
    double tot = 0;
    int nb = 0;
    std::vector<double> mean(kD);
    for (int b = 0; b < m; ++b) {
        std::fill(mean.begin(), mean.end(), 0.0);
        int n = 0;
        for (int k = 0; k < slots; ++k) {
            const int pool = members[(size_t) b * slots + k];
            if (pool < 0) continue;
            for (int t = 0; t < kPool; ++t) {
                const int key = pool * kPool + t;
                if (key >= n_keys) continue;
                for (int d = 0; d < kD; ++d) mean[d] += (double) __half2float(keys[(size_t) key * kD + d]);
                ++n;
            }
        }
        if (n == 0) continue;
        for (int d = 0; d < kD; ++d) mean[d] /= n;
        double s = 0;
        int c = 0;
        for (int k = 0; k < slots; ++k) {
            const int pool = members[(size_t) b * slots + k];
            if (pool < 0) continue;
            for (int t = 0; t < kPool; ++t) {
                const int key = pool * kPool + t;
                if (key >= n_keys) continue;
                double d2 = 0;
                for (int d = 0; d < kD; ++d) {
                    const double v = (double) __half2float(keys[(size_t) key * kD + d]) - mean[d];
                    d2 += v * v;
                }
                s += d2;
                ++c;
            }
        }
        tot += s / c;
        ++nb;
    }
    return nb ? tot / nb : 0.0;
}

double random_variance(const __half* keys, int n_keys, int m, int per, unsigned seed) {
    std::mt19937 rng(seed);
    std::vector<int> idx(n_keys);
    for (int i = 0; i < n_keys; ++i) idx[i] = i;
    std::shuffle(idx.begin(), idx.end(), rng);
    double tot = 0;
    int nb = 0;
    std::vector<double> mean(kD);
    for (int b = 0; b < m; ++b) {
        const int k0 = b * per;
        const int n = std::min(per, n_keys - k0);
        if (n <= 0) break;
        std::fill(mean.begin(), mean.end(), 0.0);
        for (int i = 0; i < n; ++i)
            for (int d = 0; d < kD; ++d) mean[d] += (double) __half2float(keys[(size_t) idx[k0 + i] * kD + d]);
        for (int d = 0; d < kD; ++d) mean[d] /= n;
        double s = 0;
        for (int i = 0; i < n; ++i) {
            double d2 = 0;
            for (int d = 0; d < kD; ++d) {
                const double v = (double) __half2float(keys[(size_t) idx[k0 + i] * kD + d]) - mean[d];
                d2 += v * v;
            }
            s += d2;
        }
        tot += s / n;
        ++nb;
    }
    return nb ? tot / nb : 0.0;
}

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// a device-wide barrier's price, so the report can bound a multi-CTA-a-block design instead of assuming it
__global__ void barrier_kernel(int n, long long* out) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const long long t0 = clock64();
    for (int i = 0; i < n; ++i) __threadfence();
    out[0] = clock64() - t0;
}

}  // namespace

int main(int argc, char** argv) {
    int blocks = 128, m = 64, cap = 0, runs = 5, iters = 20, npools = 128, oracle_n = 0;
    const char* keys_path = nullptr;
    bool smoke = false, barrier = false;
    std::string arm = "cta";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto nxt = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : nullptr; };
        if (a == "--keys") keys_path = nxt();
        else if (a == "--blocks") blocks = std::atoi(nxt());
        else if (a == "--m") m = std::atoi(nxt());
        else if (a == "--cap") cap = std::atoi(nxt());
        else if (a == "--runs") runs = std::atoi(nxt());
        else if (a == "--iters") iters = std::atoi(nxt());
        else if (a == "--pools") npools = std::atoi(nxt());
        else if (a == "--oracle") oracle_n = std::atoi(nxt());
        else if (a == "--arm") arm = nxt();
        else if (a == "--smoke") smoke = true;
        else if (a == "--barrier") barrier = true;
        else {
            std::fprintf(stderr,
                         "usage: lsa_ward_bench [--keys FILE] [--blocks N] [--m M] [--cap C]\n"
                         "                      [--runs R] [--iters I] [--pools 64|128|256]\n"
                         "                      [--arm cta|cta256|warp|warp8|recompute] [--oracle N] [--smoke]\n"
                         "                      [--barrier]\n");
            return 2;
        }
    }
    if (cap == 0) cap = std::max(16, 2 * (512 / std::max(1, m)));   // bucket_cap >= 2 * bucket_size
    const bool rec = arm == "recompute";
    const bool warp = arm == "warp" || arm == "warp8";
    const int warps = arm == "warp8" ? 8 : 4;
    const int split = (arm == "cta256") ? 2 : 1;

    ck(cudaSetDevice(0), "device");
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0), "props");
    const int nsm = prop.multiProcessorCount;

    if (barrier) {
        long long* d = nullptr;
        ck(cudaMalloc(&d, 8), "b");
        barrier_kernel<<<34, 1>>>(1000, d);
        ck(cudaDeviceSynchronize(), "bw");
        long long h = 0;
        ck(cudaMemcpy(&h, d, 8, cudaMemcpyDeviceToHost), "bh");
        std::printf("  a __threadfence in a loop: %.2f ns\n", (double) h / 1000.0);
        ck(cudaFree(d), "bf");
        return 0;
    }

    const int n_keys = npools * kPool;   // 128 pools = 512 keys; 64 pools = 256 keys.  Not independent knobs.
    const int max_blocks = std::max(std::max(blocks, std::max(1, oracle_n)), smoke ? 40 : 1);
    std::vector<__half> keys;
    if (keys_path) {
        if (!read_keys(keys_path, keys, max_blocks, n_keys)) {
            // the file is written for 512-key blocks; a 64-pool run needs its own 256-key file
            std::fprintf(stderr, "lsa_ward_bench: cannot read %s as %d blocks of %d keys%s\n", keys_path, max_blocks,
                         n_keys, npools == 64 ? " (a 256-key file is needed for --pools 64, the 512-key one is not"
                                                " a prefix)" : "");
            return 2;
        }
    } else gen_keys(keys, max_blocks, n_keys, 20261112);
    const char* src = keys_path ? "the checkpoint's own indexer projection" : "the built-in generator";

    const int smem = smem_for(npools, !rec);
    __half* d_cent = nullptr;
    int32_t *d_mem = nullptr, *d_sz = nullptr, *d_rnd = nullptr, *d_reach = nullptr;
    long long* d_phase = nullptr;
    const int slots = cap / kPool;
    ck(cudaMalloc(&d_cent, (size_t) max_blocks * m * kD * 2), "cent");
    ck(cudaMalloc(&d_mem, (size_t) max_blocks * m * slots * 4), "mem");
    ck(cudaMalloc(&d_sz, (size_t) max_blocks * m * 4), "sz");
    ck(cudaMalloc(&d_rnd, (size_t) max_blocks * 4), "rnd");
    ck(cudaMalloc(&d_reach, (size_t) max_blocks * 4), "reach");
    ck(cudaMalloc(&d_phase, 6 * sizeof(long long)), "phase");
    ck(cudaMemset(d_phase, 0, 6 * sizeof(long long)), "phase0");

    int* d_trace = nullptr;
    ck(cudaMalloc(&d_trace, 128 * sizeof(int)), "trace");
    ck(cudaMemset(d_trace, 0, 128 * sizeof(int)), "trace0");
    Params p{n_keys, npools, m, cap, 256, blocks};
    std::printf("LSA Ward: one CTA a cluster block, %d pools (%d keys, pool %d), d %d, m %d, cap %d keys, arm %s\n",
                npools, n_keys, kPool, kD, m, cap, arm.c_str());
    std::printf("  keys: %s; blocks %d; card %d SMs, 101,376 B smem a block opt-in\n", src, blocks, nsm);
    std::printf("  shared a block: %d B (%d pool sums + %d Gram + %d control) of 101,376 -> %s\n", smem,
                npools * kStride * 4, rec ? 0 : npools * (npools - 1) / 2 * 4,
                smem - npools * kStride * 4 - (rec ? 0 : npools * (npools - 1) / 2 * 4),
                smem <= 101376 ? "FITS" : "DOES NOT FIT");
    if (npools == 256 && !rec)
        std::printf("  NOTE: 256 pools with a cached Gram needs 200,192 B; running the recompute arm instead\n");

    // the dispatch, as a lambda so the timing and the correctness run take the same path
    auto launch = [&](int grid) {
        if (warp) {
            if (warps == 8) ward_warp_kernel<128, 8, true><<<grid, 8 * 32, smem>>>((const __half*) keys.data(), p, d_cent,
                                                                                 d_mem, d_sz, d_rnd, d_reach, d_phase, d_trace);
            else ward_warp_kernel<128, 4, true><<<grid, 4 * 32, smem>>>((const __half*) keys.data(), p, d_cent, d_mem,
                                                                       d_sz, d_rnd, d_reach, d_phase, d_trace);
        } else if (rec) {
            if (npools == 256)
                ward_cta_kernel<256, 256, 1, false><<<grid, 256, smem>>>((const __half*) keys.data(), p, d_cent, d_mem,
                                                                        d_sz, d_rnd, d_reach, d_phase, d_trace);
            else
                ward_cta_kernel<128, 256, 1, false><<<grid, 256, smem>>>((const __half*) keys.data(), p, d_cent, d_mem,
                                                                        d_sz, d_rnd, d_reach, d_phase, d_trace);
        } else if (npools == 64) {
            ward_cta_kernel<64, 64, 1, true><<<grid, 64, smem>>>((const __half*) keys.data(), p, d_cent, d_mem, d_sz,
                                                                   d_rnd, d_reach, d_phase, d_trace);
        } else if (split == 2) {
            ward_cta_kernel<128, 256, 2, true><<<grid, 256, smem>>>((const __half*) keys.data(), p, d_cent, d_mem,
                                                                    d_sz, d_rnd, d_reach, d_phase, d_trace);
        } else {
            ward_cta_kernel<128, 128, 1, true><<<grid, 128, smem>>>((const __half*) keys.data(), p, d_cent, d_mem,
                                                                    d_sz, d_rnd, d_reach, d_phase, d_trace);
        }
    };
    ck(cudaFuncSetAttribute(ward_cta_kernel<128, 128, 1, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem), "a1");
    ck(cudaFuncSetAttribute(ward_cta_kernel<128, 256, 2, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem), "a2");
    ck(cudaFuncSetAttribute(ward_cta_kernel<64, 64, 1, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem), "a3");
    ck(cudaFuncSetAttribute(ward_cta_kernel<128, 256, 1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem), "a4");
    ck(cudaFuncSetAttribute(ward_cta_kernel<256, 256, 1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem), "a5");
    ck(cudaFuncSetAttribute(ward_warp_kernel<128, 4, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem), "a6");
    ck(cudaFuncSetAttribute(ward_warp_kernel<128, 8, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem), "a7");

    int per_sm = 0;
    if (warp) ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, ward_warp_kernel<128, 4, true>, 4 * 32, smem), "occ");
    else if (rec) ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, ward_cta_kernel<128, 256, 1, false>, 256, smem), "occ");
    else ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, ward_cta_kernel<128, 128, 1, true>, 128, smem), "occ");

    launch(blocks);
    ck(cudaDeviceSynchronize(), "warm");

    cudaEvent_t e0, e1;
    ck(cudaEventCreate(&e0), "e");
    ck(cudaEventCreate(&e1), "e");
    auto timed = [&](int grid) {
        std::vector<double> v;
        for (int r = 0; r < runs; ++r) {
            ck(cudaEventRecord(e0), "r");
            for (int i = 0; i < iters; ++i) launch(grid);
            ck(cudaEventRecord(e1), "r1");
            ck(cudaEventSynchronize(e1), "s");
            float t = 0.f;
            ck(cudaEventElapsedTime(&t, e0, e1), "el");
            v.push_back((double) t / iters);
        }
        return median(v);
    };
    const double layer_ms = timed(blocks);   // the layer: `blocks` cluster blocks (128 at 256K)
    launch(blocks);
    ck(cudaDeviceSynchronize(), "final");
    const double one_ms = timed(1);          // one block alone: the decode case

    std::vector<int32_t> h_rnd(blocks), h_reach(blocks);
    ck(cudaMemcpy(h_rnd.data(), d_rnd, blocks * 4, cudaMemcpyDeviceToHost), "rnd");
    ck(cudaMemcpy(h_reach.data(), d_reach, blocks * 4, cudaMemcpyDeviceToHost), "reach");
    int rmin = 1 << 30, rmax = 0, rsum = 0, rfull = 0;
    for (int i = 0; i < blocks; ++i) {
        rmin = std::min(rmin, (int) h_rnd[i]);
        rmax = std::max(rmax, (int) h_rnd[i]);
        rsum += h_rnd[i];
        rfull += h_reach[i];
    }
    long long phase[6] = {0, 0, 0, 0, 0, 0};
    ck(cudaMemcpy(phase, d_phase, sizeof phase, cudaMemcpyDeviceToHost), "phase");
    const double cyc = (double) (phase[0] + phase[1] + phase[2]);

    // THE PER-LAYER COLUMN MUST USE THE LAYER'S OWN BLOCK COUNT, not the `--blocks` under test.  A 512-key block
    // is 2,048 cells so a 256K layer holds 128; a 256-key block is 1,024 cells so it holds 256.  Printing
    // `layer_ms * 12` for whatever grid was passed made the 256-key arm look 2x better than it is, because the
    // run's grid is 128 blocks either way.
    const int layer_blocks = 262144 / (n_keys * kPool);   // cells / cells-a-block, at the plan's 256K
    const double per_layer_ms = layer_ms * (double) layer_blocks / blocks;
    std::printf("\n  one block %d keys = %d cells, so a 256K layer holds %d blocks (this run is %d)\n",
                n_keys, n_keys * kPool, layer_blocks, blocks);
    std::printf("\n  %-10s %10s %10s %11s %10s\n", "measure", "us/block", "us/layer", "ms/prefill", "rounds");
    std::printf("  %-10s %10.3f %10.1f %11.3f  %d..%d, %.1f avg, %d/%d reached\n", "amortised", layer_ms * 1000.0 / blocks,
                per_layer_ms * 1000.0, per_layer_ms * 12.0, rmin, rmax, (double) rsum / blocks, rfull, blocks);
    std::printf("  %-10s %10.3f\n", "grid 1", one_ms * 1000.0);
    std::printf("  %d block(s) a SM, so the layer is %.2f waves over %d SMs\n", per_sm, (double) blocks / (nsm * per_sm), nsm);
    if (cyc > 0)
    {
        const double rc = (double) (phase[3] + phase[4]);
        std::printf("  clock64 split of block 0: pool+Gram %.0f  rounds %.0f  emit %.0f cycles (%.0f%% / %.0f%% / %.0f%%)\n",
                    (double) phase[0], (double) phase[1], (double) phase[2], 100.0 * phase[0] / cyc,
                    100.0 * phase[1] / cyc, 100.0 * phase[2] / cyc);
        std::printf("    raw clocks: %.0f %.0f %.0f | %.0f %.0f\n", (double) phase[0], (double) phase[1],
                    (double) phase[2], (double) phase[3], (double) phase[4]);
        if (rc > 0)
            std::printf("    inside the rounds: search %.0f (%.0f%%) + pick %.0f + commit %.0f; %.0f cycles a round\n",
                    (double) phase[5], 100.0 * phase[5] / (phase[5] + phase[3] + phase[4]), (double) phase[3],
                    (double) phase[4], (phase[5] + phase[3] + phase[4]) / std::max(1, rsum));
    }

    // ---- quality
    std::vector<int32_t> mem((size_t) blocks * m * slots), sz((size_t) blocks * m);
    std::vector<__half> cent((size_t) blocks * m * kD);
    ck(cudaMemcpy(mem.data(), d_mem, mem.size() * 4, cudaMemcpyDeviceToHost), "mem");
    ck(cudaMemcpy(sz.data(), d_sz, sz.size() * 4, cudaMemcpyDeviceToHost), "sz");
    ck(cudaMemcpy(cent.data(), d_cent, cent.size() * 2, cudaMemcpyDeviceToHost), "cent");
    double ward_v = 0, rand_v = 0;
    int nsz = 0, over = 0, empty = 0, hist[5] = {0, 0, 0, 0, 0};
    const int per = (n_keys + m - 1) / m;
    for (int b = 0; b < blocks; ++b) {
        ward_v += intra_variance(keys.data() + (size_t) b * n_keys * kD, n_keys, mem.data() + (size_t) b * m * slots, m, slots);
        rand_v += random_variance(keys.data() + (size_t) b * n_keys * kD, n_keys, m, per, 7u + b);
        for (int q = 0; q < m; ++q) {
            ++nsz;
            if (sz[(size_t) b * m + q] > cap) ++over;
            if (sz[(size_t) b * m + q] == 0) ++empty;
            ++hist[std::min(4, (int) sz[(size_t) b * m + q] / kPool)];
        }
    }
    ward_v /= blocks;
    rand_v /= blocks;
    std::printf("\n  quality over %d blocks: mean intra-bucket variance %.5f vs %.5f for random %d-key buckets"
                " -> %.1f%% lower\n", blocks, ward_v, rand_v, per, 100.0 * (1.0 - ward_v / rand_v));
    std::printf("  bucket sizes by pools (0/1/2/3/4): %d %d %d %d %d of %d; over cap %d, empty %d\n", hist[0], hist[1],
                hist[2], hist[3], hist[4], nsz, over, empty);

    if (oracle_n > 0) {
        int agree = 0, differ = 0, rdiff = 0;
        for (int b = 0; b < oracle_n; ++b) {
            Oracle orc;
            std::vector<__half> one_k(keys.begin() + (size_t) b * n_keys * kD, keys.begin() + (size_t) (b + 1) * n_keys * kD);
            orc.run(one_k, n_keys, npools, m, cap, 256);
            std::vector<std::vector<int>> a, c;
            for (int q = 0; q < m; ++q) {
                std::vector<int> x;
                for (int k = 0; k < slots; ++k) {
                    const int v = mem[(size_t) b * m * slots + (size_t) q * slots + k];
                    if (v >= 0) x.push_back(v);
                }
                a.push_back(x);
                std::vector<int> y;
                for (int k = 0; k < npools; ++k)
                    if (orc.label[k] == q) y.push_back(k);
                c.push_back(y);
            }
            for (auto& v : a) std::sort(v.begin(), v.end());
            for (auto& v : c) std::sort(v.begin(), v.end());
            std::sort(a.begin(), a.end());
            std::sort(c.begin(), c.end());
            if (a == c) ++agree;
            else {
                ++differ;
                if (differ == 1) {
                    std::printf("    first disagreement (block %d): kernel %zu buckets, oracle %zu buckets, rounds %d vs %d\n",
                                b, a.size(), c.size(), h_rnd[b], orc.rounds);
                    int shown = 0;
                    for (size_t q = 0; q < a.size() && shown < 4; ++q)
                        if (a[q] != c[q]) {
                            std::printf("      bucket %zu: kernel [", q);
                            for (int v : a[q]) std::printf(" %d", v);
                            std::printf(" ]  oracle [");
                            for (int v : c[q]) std::printf(" %d", v);
                            std::printf(" ]\n");
                            ++shown;
                        }
                    std::vector<int> ks(a.size(), 0), os(c.size(), 0);
                    for (size_t q = 0; q < a.size(); ++q) ks[q] = (int) a[q].size();
                    for (size_t q = 0; q < c.size(); ++q) os[q] = (int) c[q].size();
                    std::sort(ks.begin(), ks.end());
                    std::sort(os.begin(), os.end());
                    std::vector<int32_t> kt(64);
                    ck(cudaMemcpy(kt.data(), d_trace, 64 * 4, cudaMemcpyDeviceToHost), "trace");
                    std::printf("      per round (mutual, applied): kernel");
                    for (int r = 0; r < 8 && kt[2 * r + 1] != 0; ++r) std::printf(" %d/%d", kt[2 * r], kt[2 * r + 1]);
                    std::printf("\n                                   oracle");
                    for (size_t r = 0; r < orc.trace.size() / 2 && r < 8; ++r)
                        std::printf(" %d/%d", orc.trace[2 * r], orc.trace[2 * r + 1]);
                    std::printf("\n");
                    std::printf("      size histograms: kernel ");
                    for (int v : ks) std::printf("%d ", v);
                    std::printf("\n                        oracle ");
                    for (int v : os) std::printf("%d ", v);
                    std::printf("\n");
                }
            }
            if (h_reach[b] != (orc.reached ? 1 : 0)) ++rdiff;
        }
        std::printf("\n  oracle: %d of %d blocks agree block-for-block (member sets, bucket numbering free), %d differ,"
                    " %d differ on `reached`\n", agree, oracle_n, differ, rdiff);
    }

    if (smoke) {
        std::printf("\n  smoke (partial pools, invariants, reproducibility):\n");
        int bad = 0;
        const int sm = smem_bytes<128>();
        ck(cudaFuncSetAttribute(ward_cta_kernel<128, 128, 1, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm), "s0");
        for (int nb : {1, 2, 3, 5, 7, 33}) {
            for (int nk : {512, 511}) {
                Params sp{nk, (nk + kPool - 1) / kPool, m, cap, 256, nb};
                std::vector<int32_t> prev;
                for (int rep = 0; rep < 2; ++rep) {
                    ward_cta_kernel<128, 128, 1, true><<<nb, 128, sm>>>((const __half*) keys.data(), sp, d_cent, d_mem,
                                                                       d_sz, d_rnd, d_reach, d_phase, d_trace);
                    ck(cudaDeviceSynchronize(), "smoke");
                    std::vector<int32_t> mem2((size_t) nb * m * slots), sz2((size_t) nb * m);
                    std::vector<__half> c2((size_t) nb * m * kD);
                    ck(cudaMemcpy(mem2.data(), d_mem, mem2.size() * 4, cudaMemcpyDeviceToHost), "s1");
                    ck(cudaMemcpy(sz2.data(), d_sz, sz2.size() * 4, cudaMemcpyDeviceToHost), "s2");
                    ck(cudaMemcpy(c2.data(), d_cent, c2.size() * 2, cudaMemcpyDeviceToHost), "s3");
                    for (int b = 0; b < nb; ++b) {
                        std::vector<int> seen(npools, 0);
                        int sum = 0;
                        for (int q = 0; q < m; ++q) {
                            const int s = sz2[(size_t) b * m + q];
                            sum += s;
                            if (s > cap) { std::printf("    FAIL blk %d bucket %d: %d keys > cap %d\n", b, q, s, cap); ++bad; }
                            int filled = 0;
                            for (int k = 0; k < slots; ++k) {
                                const int v = mem2[(size_t) b * m * slots + (size_t) q * slots + k];
                                if (v < 0) continue;
                                if (v >= npools) { std::printf("    FAIL blk %d: pool %d out of range\n", b, v); ++bad; }
                                else if (seen[v]++) { std::printf("    FAIL blk %d: pool %d in two buckets\n", b, v); ++bad; }
                                ++filled;
                            }
                            if (s > 0 && filled == 0) { std::printf("    FAIL blk %d bucket %d: %d keys, no members\n", b, q, s); ++bad; }
                            float nrm = 0.f;
                            for (int d = 0; d < kD; ++d) {
                                const float v = (float) c2[((size_t) b * m + q) * kD + d];
                                nrm = std::fma(v, v, nrm);
                            }
                            if (s > 0 && std::fabs(std::sqrt(nrm) - 1.0f) > 2e-2f) {
                                std::printf("    FAIL blk %d bucket %d: centroid norm %.4f\n", b, q, std::sqrt(nrm));
                                ++bad;
                            }
                        }
                        int live = 0;
                        for (int q = 0; q < npools; ++q) live += seen[q];
                        if (sum != nk) { std::printf("    FAIL blk %d: sizes sum to %d, the block has %d keys\n", b, sum, nk); ++bad; }
                        if (live != (nk + kPool - 1) / kPool) {
                            std::printf("    FAIL blk %d: %d pools placed, %d live\n", b, live, (nk + kPool - 1) / kPool);
                            ++bad;
                        }
                    }
                    if (rep == 0) prev = mem2;
                    else if (prev != mem2) { std::printf("    FAIL %d blocks %d keys: not reproducible\n", nb, nk); ++bad; }
                }
            }
        }
        std::printf("    %s\n", bad ? "FAILED" : "all invariants hold: each pool in exactly one bucket, sizes sum to the"
                                                  " block's keys, none over the cap, centroids unit-norm, reproducible");
        ck(cudaFree(d_cent), "f"); ck(cudaFree(d_mem), "f"); ck(cudaFree(d_sz), "f");
        ck(cudaFree(d_rnd), "f"); ck(cudaFree(d_reach), "f"); ck(cudaFree(d_phase), "f");
        return bad ? 1 : 0;
    }
    ck(cudaFree(d_cent), "f"); ck(cudaFree(d_mem), "f"); ck(cudaFree(d_sz), "f");
    ck(cudaFree(d_rnd), "f"); ck(cudaFree(d_reach), "f"); ck(cudaFree(d_phase), "f"); ck(cudaFree(d_trace), "f");
    return 0;
}
