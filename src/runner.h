// Runner: keeps (at most one) model loaded, builds prompts from templates and runs requests.
// Shared by the REST server and the interactive `run` command.
#pragma once
#include <stdint.h>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "engine.h"
#include "store.h"

struct CompletionRequest {
    std::string model;
    // chat mode: messages; generate mode: prompt (+system/template overrides)
    std::vector<Message> messages;
    std::string prompt, system, templ;
    bool isChat = false;
    bool raw = false;
    Json options = Json::object();
    int64_t keepAliveSec = -2;      // -2 = default, -1 = forever, 0 = unload after request
    std::string suffixUnused;
};

struct RunResult {
    GenStats stats;
    int64_t totalMs = 0, loadMs = 0;
    std::string doneReason = "stop";
};

struct LoadedInfo {
    std::string name, blob;
    uint64_t size = 0;
    int64_t expiresAt = 0;   // unix seconds, 0 = never
    int contextLength = 0;
};

class Runner {
public:
    explicit Runner(Store& store);

    // Runs one completion. onText is called for each safe-to-show text piece; return false to cancel.
    // `status` receives an HTTP-like status (404 when the model is missing, 500 for other errors).
    bool complete(const CompletionRequest& req, const std::function<bool(const std::string&)>& onText,
                  RunResult& result, std::string& err, int& status);
    bool embed(const std::string& model, const std::vector<std::string>& inputs, const Json& options,
               int64_t keepAliveSec, std::vector<std::vector<float> >& out, int& tokens, std::string& err, int& status);

    std::vector<LoadedInfo> loaded();
    void unload();
    bool unloadModel(const std::string& model);
    void reap();   // unload when the keep-alive expired

    static int64_t parseKeepAlive(const Json& v, int64_t def);   // number (s) or "5m"; -2 = not given
    static int64_t defaultKeepAlive();                           // OLLAMA_KEEP_ALIVE, default 300
    // Prompt that would be sent to the engine (used by tests / `show`).
    bool buildPrompt(const ModelInfo& m, const CompletionRequest& req, Engine* eng, std::string& prompt, std::string& err);

private:
    bool ensureLoaded(const ModelInfo& m, const GenOptions& opts, int64_t keepAlive, int64_t& loadMs, std::string& err);
    void touch(int64_t keepAlive);
    GenOptions optionsFor(const ModelInfo& m, const Json& overrides);

    Store& store_;
    std::mutex mu_;        // serialises requests, like Ollama with OLLAMA_NUM_PARALLEL=1
    std::mutex stateMu_;   // guards the fields below for ps / reap
    std::unique_ptr<Engine> engine_;
    std::string loadedName_, loadedBlob_;
    uint64_t loadedSize_ = 0;
    int loadedCtx_ = 0;
    int64_t expiresAt_ = 0;
    bool busy_ = false;
};
