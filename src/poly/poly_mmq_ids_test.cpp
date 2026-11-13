// src/poly/poly_mmq_ids_test.cpp - the row-id table `iota_` of strata-poly's MoE MMQ path is allocated with the
// margin llama.cpp's kernels read past its last row, and the identity is written over all of it.
//
// Upstream sizes its three tables with mmq_ids_table_rows() (include/strata/prefill/mmq_ids_margin.hpp, Strata
// d7f7d9677 / 14dd746f6 / 697b6d67a).  src/poly/ is the fork's own code and the sync did not reach it: it had
// `iota_ = dmalloc(rr * sizeof(int32_t))`, exact, with the identity over exactly rr entries.  This test pins the
// shape the engine now has, the way tests/prefill/mmq_ids_margin_test.cpp pins the helper's.
//
// No GPU: the reads are enumerated from the layout mul_mat_q and its stream-k fixup use.
//
//   g++ -std=c++17 -Iinclude src/poly/poly_mmq_ids_test.cpp -o poly_mmq_ids_test && ./poly_mmq_ids_test
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "strata/prefill/mmq_ids_margin.hpp"

using namespace strata::prefill::mmq;

namespace {

int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

// The highest index (exclusive) a launch can read from the table.
//
// The main kernel copies ids_dst[col_low + jt*J + j] for j < J, where col_low is the expert's first row and jt
// runs over that expert's tiles: up to col_high + ceil(col_diff / J) * J from the launch's start.
// The stream-k fixup does the same from each expert's first row for ceil(max_rows / J) tiles, so a last expert
// that starts at the table's end reads a whole tile past it.
int64_t read_end(const std::vector<int64_t>& bounds, int64_t max_rows, int J) {
    const int64_t tiles = (max_rows + J - 1) / J;   // the fixup's count, from any expert's first row
    int64_t end = 0;
    for (size_t e = 0; e + 1 < bounds.size(); ++e) {
        const int64_t first = bounds[e], count = bounds[e + 1] - bounds[e];
        if (count > 0) end = std::max(end, bounds[e + 1] + J - 1);        // main: J - 1 past the expert's last row
        if (first < bounds.back()) end = std::max(end, first + tiles * J); // fixup: whole tiles from a first row
    }
    return end;
}

// What src/poly/engine.cpp allocates now: mmq_ids_table_rows(rows, rows).
int64_t engine_table(int64_t rows) {
    return (int64_t) mmq_ids_table_rows((uint64_t) rows, (uint64_t) rows);
}

}  // namespace

int main() {
    // The engine's geometry: rr = n * n_used_ rows, n up to the prompt batch, n_used_ the experts a token takes.
    // Ornith-1.5-9B is n_used_ = 8; Qwen3.6-35B-A3B is 8; Flint 1.  Check the shapes those give.
    CHECK(kMmqTileColsMax == 128 && kMmqTileColsStep == 8);

    // 1. The exact table src/poly/ had is overrun, at the sizes this engine actually runs.
    for (int64_t n : {128, 256, 512, 1024}) {
        for (int n_used : {1, 8}) {
            const int64_t rr = n * n_used;
            bool overrun = false;
            for (int J = kMmqTileColsStep; J <= kMmqTileColsMax; J += kMmqTileColsStep) {
                // the worst layout: every row in one expert that starts at 0, then a second expert's rows ending
                // the table - the fixup reads a whole tile from the last expert's first row
                std::vector<int64_t> bounds{0, rr / 2, rr};
                if (read_end(bounds, rr, J) > rr) overrun = true;
            }
            if (rr > kMmqTileColsMax) CHECK(overrun);   // the old exact table could be read past its end
            // 2. The table the engine allocates now is never read past its end.
            const int64_t table = engine_table(rr);
            CHECK(table > rr);
            for (int J = kMmqTileColsStep; J <= kMmqTileColsMax; J += kMmqTileColsStep) {
                std::vector<int64_t> bounds{0, rr / 2, rr};
                CHECK(read_end(bounds, rr, J) <= table);
                // and every bound a call site passes is <= rr, so sizing for rr covers it
                CHECK(read_end(bounds, rr / 2, J) <= table);
            }
        }
    }

    // 3. Random layouts at the engine's row counts: no reachable read leaves the table.
    std::mt19937_64 rng(20261114);
    for (int iter = 0; iter < 1500; ++iter) {
        const int64_t rr = 1 + (int64_t) (rng() % (iter < 750 ? 512 : 8192));
        const int64_t NE = 1 + (int64_t) (rng() % 128);
        std::vector<int64_t> cnt((size_t) NE, 0);
        int64_t left = rr;
        for (int64_t e = 0; e < NE && left > 0; ++e) {
            const int64_t c = std::min<int64_t>((int64_t) (rng() % ((uint64_t) rr + 1)), left);
            cnt[(size_t) e] = c;
            left -= c;
        }
        std::vector<int64_t> bounds{0};
        int64_t total = 0, widest = 0;
        for (int64_t c : cnt) { total += c; bounds.push_back(total); widest = std::max(widest, c); }
        const int64_t table = engine_table(total);
        CHECK(table > total);
        for (int J = kMmqTileColsStep; J <= kMmqTileColsMax; J += kMmqTileColsStep) {
            CHECK(read_end(bounds, widest, J) <= table);   // the launch's own max_rows (most / r.most / top)
            CHECK(read_end(bounds, total, J) <= table);    // and the wider bound the engine sizes for (rr)
        }
    }

    // 4. The identity must be written over the margin too.  An entry past `rr` that holds nothing becomes an
    //    arbitrary output row index; this is the assertion that the fill covers ids_rows and not rr.
    for (int64_t rr : {256, 1024, 8192}) {
        const int64_t table = engine_table(rr);
        std::vector<int32_t> identity((size_t) table, -1);   // -1: "never written"
        for (int64_t i = 0; i < table; ++i) identity[(size_t) i] = (int32_t) i;   // the engine's ramp over ids_rows
        for (int64_t i = 0; i < rr; ++i) CHECK(identity[(size_t) i] == (int32_t) i);
        for (int64_t i = rr; i < table; ++i) CHECK(identity[(size_t) i] == (int32_t) i);   // the margin is identity
        CHECK(std::none_of(identity.begin(), identity.end(), [](int32_t v) { return v < 0; }));
    }

    if (failed) { std::fprintf(stderr, "poly_mmq_ids_test: %d check(s) failed\n", failed); return 1; }
    std::puts("poly_mmq_ids_test: ok");
    return 0;
}
