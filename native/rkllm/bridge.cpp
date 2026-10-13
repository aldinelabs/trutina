// Build against the official RKLLM 1.3.1 header, never duplicate its ABI in Python.
#include <cstddef>
#include "rkllm.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <new>
#include <string>

struct Context {
    LLMHandle handle = nullptr;
    std::string model_path;
    RKLLMParam param{};
    RKLLMCallback callbacks{};
    std::mutex mutex;
    const int32_t* labels = nullptr;
    float* output = nullptr;
    int count = 0;
    bool received = false;
    std::string error;
};

static int result_callback(RKLLMResult* result, void* userdata, LLMCallState state) {
    auto* ctx = static_cast<Context*>(userdata);
    if (state == RKLLM_RUN_ERROR) ctx->error = "RKLLM callback reported an inference error";
    if (!result || !result->logits.logits || result->logits.num_tokens <= 0) return 0;
    const auto& logits = result->logits;
    if (logits.vocab_size <= 0) {
        ctx->error = "Invalid RKLLM vocabulary size";
        return 2;
    }
    const float* last = logits.logits +
        static_cast<size_t>(logits.num_tokens - 1) * logits.vocab_size;
    for (int i = 0; i < ctx->count; ++i) {
        if (ctx->labels[i] < 0 || ctx->labels[i] >= logits.vocab_size) {
            ctx->error = "Option token is outside the RKLLM vocabulary";
            return 2;
        }
        ctx->output[i] = last[ctx->labels[i]];
        if (!std::isfinite(ctx->output[i])) ctx->error = "Non-finite RKLLM logits";
    }
    ctx->received = true;
    // Copy only the 26 decision logits before releasing the potentially huge buffer.
    return 2;
}

extern "C" {
int trutina_rkllm_open(const char* path, int context_len, void** out) {
    if (!path || !out || context_len <= 0) return -1;
    *out = nullptr;
    auto* ctx = new (std::nothrow) Context;
    if (!ctx) return -1;
    ctx->model_path = path;
    auto& param = ctx->param;
    param = rkllm_createDefaultParam();
    param.model_path = ctx->model_path.c_str();
    param.max_context_len = context_len;
    param.max_new_tokens = 1;
    param.is_async = false;
    param.temperature = 1;
    param.repeat_penalty = 1;
    param.frequency_penalty = 0;
    param.presence_penalty = 0;
    param.extend_param.n_batch = 1;
    auto& callbacks = ctx->callbacks;
    callbacks.result_callback = result_callback;
    callbacks.result_userdata = ctx;
    int rc = rkllm_init(&ctx->handle, &param, &callbacks);
    if (rc == 0) rc = rkllm_set_chat_template(ctx->handle, "", "", "");
    if (rc != 0) {
        if (ctx->handle) rkllm_destroy(ctx->handle);
        delete ctx;
        return rc;
    }
    *out = ctx;
    return 0;
}

int trutina_rkllm_logits(void* opaque, const int32_t* tokens, size_t n_tokens,
                        const int32_t* labels, int count, float* output) {
    if (!opaque || !tokens || !n_tokens || !labels || count <= 0 || !output) return -1;
    auto* ctx = static_cast<Context*>(opaque);
    std::lock_guard<std::mutex> guard(ctx->mutex);
    ctx->error.clear();
    ctx->received = false;
    ctx->labels = labels;
    ctx->output = output;
    ctx->count = count;
    int rc = rkllm_clear_kv_cache(ctx->handle, 0, nullptr, nullptr);
    if (rc != 0) {
        ctx->error = "Could not clear RKLLM cache before request";
        return rc;
    }
    RKLLMInput input{};
    input.input_type = RKLLM_INPUT_TOKEN;
    input.token_input.input_ids = const_cast<int32_t*>(tokens);
    input.token_input.n_tokens = n_tokens;
    input.enable_thinking = false;
    RKLLMInferParam infer{};
    infer.mode = RKLLM_INFER_GET_LOGITS;
    infer.keep_history = 0;
    rc = rkllm_run(ctx->handle, &input, &infer, ctx);
    ctx->labels = nullptr;
    ctx->output = nullptr;
    if (rc != 0 && ctx->error.empty()) ctx->error = "rkllm_run failed: " + std::to_string(rc);
    if (!ctx->received && ctx->error.empty()) ctx->error = "RKLLM returned no logits";
    return !ctx->error.empty() ? -1 : rc;
}

const char* trutina_rkllm_error(void* opaque) {
    return opaque ? static_cast<Context*>(opaque)->error.c_str() : "Invalid RKLLM context";
}

void trutina_rkllm_close(void* opaque) {
    if (!opaque) return;
    auto* ctx = static_cast<Context*>(opaque);
    rkllm_destroy(ctx->handle);
    delete ctx;
}
}
