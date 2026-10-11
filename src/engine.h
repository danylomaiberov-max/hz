// Inference engine abstraction. The real backend wraps llama.cpp (engine_llama.cpp);
// a deterministic mock (engine_mock.cpp) is used for testing the CLI/API without weights.
#pragma once
#include <stdint.h>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "json.h"
#include "store.h"

struct GenOptions {
    int num_ctx = 2048;
    int num_predict = -1;        // -1 = until EOS / context full
    int top_k = 40;
    float top_p = 0.9f;
    float min_p = 0.0f;
    float temperature = 0.8f;
    float repeat_penalty = 1.1f;
    int repeat_last_n = 64;
    int seed = -1;               // -1 = random
    int num_thread = 0;          // 0 = auto
    int num_batch = 256;
    std::vector<std::string> stop;

    // Applies recognized keys of `j` (Ollama "options" object) on top of the current values.
    void apply(const Json& j);
};

struct GenStats {
    int promptTokens = 0, evalTokens = 0;
    int64_t loadMs = 0, promptMs = 0, evalMs = 0;
    std::string doneReason = "stop";   // "stop" | "length"
};

class Engine {
public:
    virtual ~Engine() {}
    // Loads weights; num_ctx fixes the KV size. Returns false with err on failure (e.g. out of memory).
    virtual bool load(const std::string& path, const GenOptions& opts, std::string& err) = 0;
    // Streams text pieces to onText (return false to abort). `prompt` is already fully templated.
    virtual bool generate(const std::string& prompt, const GenOptions& opts, bool addSpecial,
                          const std::function<bool(const std::string&)>& onText, GenStats& stats,
                          std::string& err) = 0;
    virtual bool embed(const std::vector<std::string>& inputs, std::vector<std::vector<float> >& out,
                       int& tokens, std::string& err) = 0;
    // Applies a jinja-style chat template from GGUF metadata via the backend; false if unsupported.
    virtual bool applyChatTemplate(const std::string& tmpl, const std::vector<Message>& msgs, bool addAssistant,
                                   std::string& out) = 0;
    virtual int contextSize() const = 0;
    virtual std::string name() const = 0;
};

std::unique_ptr<Engine> createEngine();   // llama.cpp if compiled in, otherwise the mock
bool engineIsMock();

// Holds back text that might be the beginning of a stop sequence.
class StopFilter {
public:
    explicit StopFilter(const std::vector<std::string>& stops) : stops_(stops), stopped_(false) {}
    // Returns text that is safe to emit now.
    std::string feed(const std::string& piece);
    std::string flush();   // at end of generation: release held-back text (if not stopped)
    bool stopped() const { return stopped_; }
private:
    std::vector<std::string> stops_;
    std::string pending_;
    bool stopped_;
};

// Number of bytes at the front of `s` that form complete UTF-8 sequences.
size_t utf8CompleteLen(const std::string& s);
