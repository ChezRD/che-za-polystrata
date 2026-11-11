// Fork-only guard: the upstream-file hooks this tree depends on. A merge that drops one fails here,
// instead of surfacing when a user loads the 12G or REAP-320 model.
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cstdio>

int main() {
    int bad = 0;
    const auto check = [&](bool ok, const char* what) {
        if (!ok) {
            std::fprintf(stderr, "local_hooks_test: %s\n", what);
            ++bad;
        }
    };
    check(strata::kernels::native_mmvq_supported(19), "IQ1_S (type 19) is missing from native_mmvq_supported");
    check(strata::kernels::iq_row_bytes(19, 256) > 0, "IQ1_S (type 19) has no row bytes");
    check(strata::kernels::native_expert_supported(20, 20, 2560, 640),
          "IQ4_NL (type 20) gate/up is not accepted at 2560x640");
    if (bad) return 1;
    std::printf("local_hooks_test: ok\n");
    return 0;
}
