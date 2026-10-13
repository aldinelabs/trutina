// Hardware-free contract test compiled against the pinned official SDK header.
#include "bridge.cpp"
#include <cassert>
#include <limits>
#include <vector>

static RKLLMCallback callbacks;
static int clears = 0;
static bool inject_nan = false;
static bool no_output = false;
extern "C" {
RKLLMParam rkllm_createDefaultParam() { return RKLLMParam{}; }
int rkllm_init(LLMHandle* handle, RKLLMParam* param, RKLLMCallback* cb) {
    assert(!param->is_async && param->max_new_tokens == 1);
    callbacks = *cb;
    *handle = reinterpret_cast<void*>(1);
    return 0;
}
int rkllm_set_chat_template(LLMHandle, const char* a, const char* b, const char* c) {
    assert(!*a && !*b && !*c);
    return 0;
}
int rkllm_destroy(LLMHandle) { return 0; }
int rkllm_clear_kv_cache(LLMHandle, int keep, int* begin, int* end) {
    assert(!keep && !begin && !end);
    ++clears;
    return 0;
}
int rkllm_run(LLMHandle, RKLLMInput* input, RKLLMInferParam* infer, void* userdata) {
    assert(input->input_type == RKLLM_INPUT_TOKEN);
    assert(infer->mode == RKLLM_INFER_GET_LOGITS && !infer->keep_history);
    if (no_output) return 0;
    std::vector<float> buffer(200, -100);
    for (int i = 0; i < 100; ++i) buffer[100+i] = static_cast<float>(i);
    if (inject_nan) buffer[165] = std::numeric_limits<float>::quiet_NaN();
    RKLLMResult result{};
    result.logits = {buffer.data(), 100, 2};
    assert(callbacks.result_callback(&result, userdata, RKLLM_RUN_NORMAL) == 2);
    std::fill(buffer.begin(), buffer.end(), 0); // copied results must survive buffer release
    callbacks.result_callback(nullptr, userdata, RKLLM_RUN_FINISH);
    return 0;
}
}

int main() {
    void* ctx = nullptr;
    assert(trutina_rkllm_open("fake.rkllm", 4096, &ctx) == 0);
    int32_t tokens[] = {1, 2, 3}, labels[] = {65, 66};
    float output[2]{};
    for (int i = 0; i < 2; ++i) {
        assert(trutina_rkllm_logits(ctx, tokens, 3, labels, 2, output) == 0);
        assert(output[0] == 65 && output[1] == 66);
    }
    assert(clears == 2);
    inject_nan = true;
    assert(trutina_rkllm_logits(ctx, tokens, 3, labels, 2, output) != 0);
    assert(std::string(trutina_rkllm_error(ctx)).find("Non-finite") != std::string::npos);
    inject_nan = false;
    no_output = true;
    assert(trutina_rkllm_logits(ctx, tokens, 3, labels, 2, output) != 0);
    assert(std::string(trutina_rkllm_error(ctx)).find("no logits") != std::string::npos);
    trutina_rkllm_close(ctx);
}
