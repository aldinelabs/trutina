// RK3588 INT8 Linear primitives. CPU owns the decoder's execution order.
// Build with the official RKNN headers; Python sees only this narrow ABI.
#include "rknn_matmul_api.h"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <map>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

static void check(int rc, const char* operation) {
    if (rc != 0) throw std::runtime_error(std::string(operation) + ": " + std::to_string(rc));
}
using Clock = std::chrono::steady_clock;
static uint64_t elapsed(Clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count();
}
struct Kernel {
    rknn_matmul_ctx context = 0;
    rknn_matmul_info info{};
    rknn_matmul_io_attr io{};
    rknn_tensor_mem *a = nullptr, *c = nullptr;
    ~Kernel() {
        if (a) rknn_destroy_mem(context, a);
        if (c) rknn_destroy_mem(context, c);
        if (context) rknn_matmul_destroy(context);
    }
};
struct Pool {
    int chunk = 64, cores = 1, live = 0;
    uint64_t calls = 0;
    uint64_t bind_ns = 0, quant_ns = 0, run_ns = 0, output_ns = 0;
    std::mutex mutex;
    std::string error;
    std::map<std::pair<int, int>, std::shared_ptr<Kernel>> kernels;
    std::shared_ptr<Kernel> get(int k, int n) {
        const auto key = std::make_pair(k, n);
        auto found = kernels.find(key);
        if (found != kernels.end()) return found->second;
        auto result = std::make_shared<Kernel>();
        auto& info = result->info;
        info.M = chunk; info.K = k; info.N = n;
        info.type = RKNN_INT8_MM_INT8_TO_INT32;
        info.B_layout = 1; info.AC_layout = 0;
        check(rknn_matmul_create(&result->context, &info, &result->io), "matmul create");
        check(rknn_matmul_set_core_mask(result->context, static_cast<rknn_core_mask>(cores)), "core mask");
        result->a = rknn_create_mem(result->context, result->io.A.size);
        result->c = rknn_create_mem(result->context, result->io.C.size);
        if (!result->a || !result->c) throw std::runtime_error("NPU IO allocation failed");
        if (result->io.A.size < static_cast<size_t>(chunk)*k ||
            result->io.B.size < static_cast<size_t>(k)*n ||
            result->io.C.size < static_cast<size_t>(chunk)*n*sizeof(int32_t))
            throw std::runtime_error("Unexpected NPU buffer sizes");
        check(rknn_matmul_set_io_mem(result->context, result->a, &result->io.A), "bind A");
        check(rknn_matmul_set_io_mem(result->context, result->c, &result->io.C), "bind C");
        kernels.emplace(key, result);
        return result;
    }
};
struct Linear {
    Pool* pool = nullptr;
    std::shared_ptr<Kernel> kernel;
    rknn_tensor_mem* b = nullptr;
    std::vector<float> scales;
    int k = 0, n = 0;
    ~Linear() { if (b) rknn_destroy_mem(kernel->context, b); }
};
extern "C" {
int trutina_mm_open(int chunk, int cores, void** out) {
    if (!out || chunk < 1 || chunk > 512 || cores < 1 || cores > 7) return -1;
    *out = nullptr;
    try { auto* p = new Pool; p->chunk = chunk; p->cores = cores; *out = p; return 0; }
    catch (...) { return -1; }
}
int trutina_mm_linear(void* opaque, const int8_t* weights_kn, const float* scales,
                      int k, int n, void** out) {
    if (!opaque || !out) return -1;
    *out = nullptr;
    auto* pool = static_cast<Pool*>(opaque);
    std::lock_guard<std::mutex> guard(pool->mutex);
    pool->error.clear();
    try {
        if (!weights_kn || !scales || k <= 0 || k > 10240 || k % 32 ||
            n <= 0 || n > 32768 || n % 32) throw std::runtime_error("Unsupported Linear shape");
        for (int i = 0; i < n; ++i)
            if (!std::isfinite(scales[i]) || scales[i] <= 0) throw std::runtime_error("Invalid weight scale");
        auto linear = std::make_unique<Linear>();
        linear->pool = pool; linear->k = k; linear->n = n;
        linear->scales.assign(scales, scales + n);
        linear->kernel = pool->get(k, n);
        auto& kernel = *linear->kernel;
        linear->b = rknn_create_mem(kernel.context, kernel.io.B.size);
        if (!linear->b) throw std::runtime_error("NPU weight allocation failed");
        std::memset(linear->b->virt_addr, 0, kernel.io.B.size);
        check(rknn_B_normal_layout_to_native_layout(const_cast<int8_t*>(weights_kn),
                  linear->b->virt_addr, k, n, &kernel.info), "pack B");
        check(rknn_mem_sync(kernel.context, linear->b, RKNN_MEMORY_SYNC_TO_DEVICE), "sync B");
        *out = linear.release(); ++pool->live;
        return 0;
    } catch (const std::exception& e) { pool->error = e.what(); return -1; }
    catch (...) { pool->error = "Unknown Linear initialization error"; return -1; }
}
int trutina_mm_forward(void* opaque, const float* input, size_t rows, float* output) {
    if (!opaque) return -1;
    auto* linear = static_cast<Linear*>(opaque);
    auto* pool = linear->pool;
    std::lock_guard<std::mutex> guard(pool->mutex);
    pool->error.clear();
    try {
        if (!input || !output || rows == 0 || rows > 65536) throw std::runtime_error("Invalid input rows");
        auto& kernel = *linear->kernel;
        auto started = Clock::now();
        check(rknn_matmul_set_io_mem(kernel.context, linear->b, &kernel.io.B), "bind B");
        pool->bind_ns += elapsed(started);
        auto* a = static_cast<int8_t*>(kernel.a->virt_addr);
        const auto* c = static_cast<const int32_t*>(kernel.c->virt_addr);
        std::vector<float> row_scales(pool->chunk);
        for (size_t offset = 0; offset < rows; offset += pool->chunk) {
            const size_t take = std::min(static_cast<size_t>(pool->chunk), rows-offset);
            started = Clock::now();
            std::memset(a, 0, kernel.io.A.size);
            for (size_t r = 0; r < take; ++r) {
                const float* src = input + (offset+r)*linear->k;
                float maximum = 0;
                for (int i = 0; i < linear->k; ++i) {
                    if (!std::isfinite(src[i])) throw std::runtime_error("Non-finite input");
                    maximum = std::max(maximum, std::abs(src[i]));
                }
                const float scale = maximum > 0 ? std::max(maximum / 127.f, std::numeric_limits<float>::min()) : 1.f;
                row_scales[r] = scale;
                for (int i = 0; i < linear->k; ++i)
                    a[r*linear->k+i] = static_cast<int8_t>(std::clamp(std::nearbyint(src[i]/scale), -127.f, 127.f));
            }
            check(rknn_mem_sync(kernel.context, kernel.a, RKNN_MEMORY_SYNC_TO_DEVICE), "sync A");
            pool->quant_ns += elapsed(started);
            started = Clock::now();
            check(rknn_matmul_run(kernel.context), "matmul run");
            check(rknn_mem_sync(kernel.context, kernel.c, RKNN_MEMORY_SYNC_FROM_DEVICE), "sync C");
            pool->run_ns += elapsed(started);
            ++pool->calls;
            started = Clock::now();
            for (size_t r = 0; r < take; ++r)
                for (int j = 0; j < linear->n; ++j) {
                    const float value = c[r*linear->n+j] * row_scales[r] * linear->scales[j];
                    if (!std::isfinite(value)) throw std::runtime_error("Non-finite output");
                    output[(offset+r)*linear->n+j] = value;
                }
            pool->output_ns += elapsed(started);
        }
        return 0;
    } catch (const std::exception& e) { pool->error = e.what(); return -1; }
    catch (...) { pool->error = "Unknown matmul inference error"; return -1; }
}
const char* trutina_mm_error(void* opaque) {
    return opaque ? static_cast<Pool*>(opaque)->error.c_str() : "Invalid matmul pool";
}
uint64_t trutina_mm_calls(void* opaque) { return opaque ? static_cast<Pool*>(opaque)->calls : 0; }
int trutina_mm_stats(void* opaque, uint64_t* output) {
    if (!opaque || !output) return -1;
    auto* pool = static_cast<Pool*>(opaque);
    std::lock_guard<std::mutex> guard(pool->mutex);
    output[0] = pool->calls; output[1] = pool->bind_ns; output[2] = pool->quant_ns;
    output[3] = pool->run_ns; output[4] = pool->output_ns;
    return 0;
}
int trutina_mm_free(void* opaque) {
    if (!opaque) return 0;
    auto* linear = static_cast<Linear*>(opaque);
    auto* pool = linear->pool;
    std::lock_guard<std::mutex> guard(pool->mutex);
    delete linear; --pool->live; return 0;
}
int trutina_mm_close(void* opaque) {
    if (!opaque) return 0;
    auto* pool = static_cast<Pool*>(opaque);
    if (pool->live) { pool->error = "Close all Linear handles first"; return -1; }
    delete pool; return 0;
}
}
