// RK3588 INT8 Linear primitives. CPU owns the decoder's execution order.
// Build with the official RKNN headers; Python sees only this narrow ABI.
#include "rknn_matmul_api.h"
#include <arm_neon.h>
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
// Per-token activation quantization and dequantization in NEON. Same operations
// as the scalar form (IEEE divide, round-to-nearest-even, multiply order), so
// results are bit-identical; FMAX propagates NaN into the finiteness checks.
// K and tile widths are multiples of 32.
static float abs_max(const float* x, int k) {
    float32x4_t m0 = vdupq_n_f32(0), m1 = m0;
    for (int i = 0; i < k; i += 8) {
        m0 = vmaxq_f32(m0, vabsq_f32(vld1q_f32(x+i)));
        m1 = vmaxq_f32(m1, vabsq_f32(vld1q_f32(x+i+4)));
    }
    return vmaxvq_f32(vmaxq_f32(m0, m1));
}
static void quantize_row(const float* x, int k, float scale, int8_t* out) {
    const float32x4_t s = vdupq_n_f32(scale);
    const int32x4_t lo = vdupq_n_s32(-127), hi = vdupq_n_s32(127);
    for (int i = 0; i < k; i += 16) {
        int32x4_t q[4];
        for (int j = 0; j < 4; ++j)
            q[j] = vmaxq_s32(lo, vminq_s32(hi, vcvtnq_s32_f32(vdivq_f32(vld1q_f32(x+i+4*j), s))));
        const int16x8_t a = vcombine_s16(vqmovn_s32(q[0]), vqmovn_s32(q[1]));
        const int16x8_t b = vcombine_s16(vqmovn_s32(q[2]), vqmovn_s32(q[3]));
        vst1q_s8(out+i, vcombine_s8(vqmovn_s16(a), vqmovn_s16(b)));
    }
}
// Returns the largest |output| (NaN if any output is NaN).
static float dequantize_row(const int32_t* c, int n, float row_scale, const float* scales, float* out) {
    const float32x4_t r = vdupq_n_f32(row_scale);
    float32x4_t m = vdupq_n_f32(0);
    for (int j = 0; j < n; j += 4) {
        const float32x4_t v = vmulq_f32(vmulq_f32(vcvtq_f32_s32(vld1q_s32(c+j)), r), vld1q_f32(scales+j));
        vst1q_f32(out+j, v);
        m = vmaxq_f32(m, vabsq_f32(v));
    }
    return vmaxvq_f32(m);
}
// One NPU kernel per tile. Measured on RK3588 (M=64): K=9216 runs in 135 ms but
// three K=3072 slices take 0.9 ms each; K=4096 is also slower than 2x2048 and
// N=9216 slower than 2x4608. K slices accumulate in INT32, so tiling is exact.
constexpr int kMaxTileK = 3072, kMaxTileN = 4608;
static int tile_size(int size, int limit) {
    const int parts = (size + limit - 1) / limit;
    return ((size + parts - 1) / parts + 31) / 32 * 32;
}
struct Tile {
    int k0 = 0, k = 0, n0 = 0, n = 0;
    std::shared_ptr<Kernel> kernel;
    rknn_tensor_mem* b = nullptr;
    ~Tile() { if (b) rknn_destroy_mem(kernel->context, b); }
};
struct Linear {
    Pool* pool = nullptr;
    std::vector<std::unique_ptr<Tile>> tiles;
    std::vector<float> scales;
    int k = 0, n = 0;
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
        const int tile_k = tile_size(k, kMaxTileK), tile_n = tile_size(n, kMaxTileN);
        std::vector<int8_t> block;
        for (int n0 = 0; n0 < n; n0 += tile_n)
            for (int k0 = 0; k0 < k; k0 += tile_k) {
                auto tile = std::make_unique<Tile>();
                tile->k0 = k0; tile->k = std::min(tile_k, k-k0);
                tile->n0 = n0; tile->n = std::min(tile_n, n-n0);
                tile->kernel = pool->get(tile->k, tile->n);
                auto& kernel = *tile->kernel;
                tile->b = rknn_create_mem(kernel.context, kernel.io.B.size);
                if (!tile->b) throw std::runtime_error("NPU weight allocation failed");
                std::memset(tile->b->virt_addr, 0, kernel.io.B.size);
                block.resize(static_cast<size_t>(tile->k)*tile->n);
                for (int i = 0; i < tile->k; ++i)
                    std::memcpy(&block[static_cast<size_t>(i)*tile->n],
                                weights_kn + static_cast<size_t>(k0+i)*n + n0, tile->n);
                check(rknn_B_normal_layout_to_native_layout(block.data(),
                          tile->b->virt_addr, tile->k, tile->n, &kernel.info), "pack B");
                check(rknn_mem_sync(kernel.context, tile->b, RKNN_MEMORY_SYNC_TO_DEVICE), "sync B");
                linear->tiles.push_back(std::move(tile));
            }
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
        const int k = linear->k, n = linear->n;
        std::vector<float> row_scales(pool->chunk);
        std::vector<int8_t> quantized(static_cast<size_t>(pool->chunk)*k);
        std::vector<int32_t> sum;
        for (size_t offset = 0; offset < rows; offset += pool->chunk) {
            const size_t take = std::min(static_cast<size_t>(pool->chunk), rows-offset);
            auto started = Clock::now();
            for (size_t r = 0; r < take; ++r) {
                const float* src = input + (offset+r)*k;
                const float maximum = abs_max(src, k);
                if (!std::isfinite(maximum)) throw std::runtime_error("Non-finite input");
                const float scale = maximum > 0 ? std::max(maximum / 127.f, std::numeric_limits<float>::min()) : 1.f;
                row_scales[r] = scale;
                quantize_row(src, k, scale, &quantized[r*k]);
            }
            pool->quant_ns += elapsed(started);
            for (size_t t = 0; t < linear->tiles.size(); ++t) {
                const auto& tile = *linear->tiles[t];
                auto& kernel = *tile.kernel;
                const bool first = tile.k0 == 0, last = tile.k0 + tile.k == k;
                started = Clock::now();
                check(rknn_matmul_set_io_mem(kernel.context, tile.b, &kernel.io.B), "bind B");
                pool->bind_ns += elapsed(started);
                started = Clock::now();
                auto* a = static_cast<int8_t*>(kernel.a->virt_addr);
                std::memset(a, 0, kernel.io.A.size);
                for (size_t r = 0; r < take; ++r)
                    std::memcpy(a + r*tile.k, &quantized[r*k + tile.k0], tile.k);
                check(rknn_mem_sync(kernel.context, kernel.a, RKNN_MEMORY_SYNC_TO_DEVICE), "sync A");
                pool->quant_ns += elapsed(started);
                started = Clock::now();
                check(rknn_matmul_run(kernel.context), "matmul run");
                check(rknn_mem_sync(kernel.context, kernel.c, RKNN_MEMORY_SYNC_FROM_DEVICE), "sync C");
                pool->run_ns += elapsed(started);
                ++pool->calls;
                started = Clock::now();
                const auto* c = static_cast<const int32_t*>(kernel.c->virt_addr);
                const size_t width = tile.n;
                if (first && !last) sum.assign(c, c + take*width);
                else if (!first)
                    for (size_t i = 0; i < take*width; ++i) sum[i] += c[i];
                if (last) {
                    const int32_t* totals = first ? c : sum.data();
                    for (size_t r = 0; r < take; ++r) {
                        const float largest = dequantize_row(totals + r*width, tile.n, row_scales[r],
                            &linear->scales[tile.n0], output + (offset+r)*n + tile.n0);
                        if (!std::isfinite(largest)) throw std::runtime_error("Non-finite output");
                    }
                }
                pool->output_ns += elapsed(started);
            }
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
