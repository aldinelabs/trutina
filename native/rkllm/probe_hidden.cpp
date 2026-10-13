// Experimental diagnostic CLI; does not change the serving bridge.
// probe_hidden MODEL INPUT.f32 OUTPUT.f32 TOKENS HIDDEN CONTEXT
#include <cstddef>
#include "rkllm.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

struct Capture {
    size_t tokens = 0, hidden = 0;
    bool received = false;
    std::string error;
    std::vector<float> output;
};
static int capture(RKLLMResult* result, void* userdata, LLMCallState state) {
    auto* c = static_cast<Capture*>(userdata);
    if (state == RKLLM_RUN_ERROR) c->error = "Runtime callback error";
    if (!result || !result->last_hidden_layer.hidden_states) return 0;
    const auto& h = result->last_hidden_layer;
    if (h.num_tokens != static_cast<int>(c->tokens) ||
        h.embd_size != static_cast<int>(c->hidden) || c->received) {
        c->error = "Expected exactly one full-sequence hidden-state output";
        return 2;
    }
    const auto n = c->tokens * c->hidden;
    c->output.assign(h.hidden_states, h.hidden_states + n);
    c->received = true;
    if (!std::all_of(c->output.begin(), c->output.end(),
                     [](float v) { return std::isfinite(v); })) c->error = "Non-finite hidden state";
    return 2; // All values copied before the SDK releases its buffer.
}
int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "Usage: probe_hidden MODEL INPUT.f32 OUTPUT.f32 TOKENS HIDDEN CONTEXT\n";
        return 2;
    }
    Capture c;
    int context = 0;
    try {
        c.tokens = std::stoul(argv[4]); c.hidden = std::stoul(argv[5]);
        context = std::stoi(argv[6]);
    } catch (...) { return 2; }
    // Bounds protect both the diagnostic input and callback allocation.
    if (c.tokens == 0 || c.tokens > 16384 || c.hidden == 0 || c.hidden > 16384 ||
        context <= 0 || c.tokens >= static_cast<size_t>(context)) return 2;
    const size_t bytes = c.tokens * c.hidden * sizeof(float);
    std::ifstream in(argv[2], std::ios::binary | std::ios::ate);
    if (!in || in.tellg() != static_cast<std::streamoff>(bytes)) return 2;
    in.seekg(0);
    std::vector<float> input(c.tokens * c.hidden);
    if (!in.read(reinterpret_cast<char*>(input.data()), bytes)) return 2;
    if (!std::all_of(input.begin(), input.end(), [](float v) { return std::isfinite(v); })) return 2;
    RKLLMParam param = rkllm_createDefaultParam();
    param.model_path = argv[1]; param.max_context_len = context;
    param.max_new_tokens = 1; param.is_async = false;
    param.extend_param.n_batch = 1;
    RKLLMCallback callbacks{};
    callbacks.result_callback = capture; callbacks.result_userdata = &c;
    LLMHandle handle = nullptr;
    int rc = rkllm_init(&handle, &param, &callbacks);
    if (rc == 0) rc = rkllm_set_chat_template(handle, "", "", "");
    if (rc == 0) rc = rkllm_clear_kv_cache(handle, 0, nullptr, nullptr);
    RKLLMInput request{};
    request.input_type = RKLLM_INPUT_EMBED;
    request.embed_input.embed = input.data(); request.embed_input.n_tokens = c.tokens;
    request.enable_thinking = false;
    RKLLMInferParam infer{};
    infer.mode = RKLLM_INFER_GET_LAST_HIDDEN_LAYER; infer.keep_history = 0;
    if (rc == 0) rc = rkllm_run(handle, &request, &infer, &c);
    if (handle) rkllm_destroy(handle);
    if (rc != 0 || !c.received || !c.error.empty()) {
        std::cerr << "Hidden-state probe failed: rc=" << rc << " " << c.error << "\n";
        return 1;
    }
    std::ofstream out(argv[3], std::ios::binary | std::ios::trunc);
    if (!out.write(reinterpret_cast<const char*>(c.output.data()), bytes)) return 1;
    std::cout << "TRUTINA_HIDDEN_OK tokens=" << c.tokens << " hidden=" << c.hidden << "\n";
    return 0;
}
