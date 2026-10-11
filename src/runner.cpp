#include "runner.h"

#include <cstdlib>

#include "template.h"
#include "util.h"

Runner::Runner(Store& store) : store_(store) {}

int64_t Runner::defaultKeepAlive() {
    std::string e = util::getEnv("OLLAMA_KEEP_ALIVE");
    if (e.empty()) return 300;
    Json j;
    if (Json::parse(e, j)) return parseKeepAlive(j, 300);
    return parseKeepAlive(Json(e), 300);
}

int64_t Runner::parseKeepAlive(const Json& v, int64_t def) {
    if (v.isNull()) return def;
    if (v.isNumber()) return v.num() < 0 ? -1 : (int64_t)v.num();
    if (v.isString()) {
        std::string s = util::trim(v.str());
        if (s.empty()) return def;
        char* end = 0; double n = std::strtod(s.c_str(), &end);
        std::string unit = end ? util::lower(util::trim(end)) : "";
        double mult = 1;
        if (unit == "ms") mult = 0.001; else if (unit == "s" || unit.empty()) mult = 1;
        else if (unit == "m") mult = 60; else if (unit == "h") mult = 3600; else return def;
        return n < 0 ? -1 : (int64_t)(n * mult);
    }
    return def;
}

GenOptions Runner::optionsFor(const ModelInfo& m, const Json& overrides) {
    GenOptions o;
    std::string ctx = util::getEnv("OLLAMA_CONTEXT_LENGTH");
    if (!ctx.empty()) o.num_ctx = std::atoi(ctx.c_str());
    std::string thr = util::getEnv("OLLAMA_NUM_THREADS");
    if (!thr.empty()) o.num_thread = std::atoi(thr.c_str());
    o.apply(m.params);
    o.apply(overrides);
    // Cap the context to what the model was trained for.
    if (m.gguf.valid && m.gguf.contextLength > 0 && (uint64_t)o.num_ctx > m.gguf.contextLength)
        o.num_ctx = (int)m.gguf.contextLength;
    return o;
}

bool Runner::ensureLoaded(const ModelInfo& m, const GenOptions& opts, int64_t keepAlive, int64_t& loadMs, std::string& err) {
    loadMs = 0;
    {
        std::lock_guard<std::mutex> g(stateMu_);
        if (engine_ && loadedName_ == m.name.full() && loadedBlob_ == m.blob && loadedCtx_ == opts.num_ctx) {
            busy_ = true; (void)keepAlive; return true;
        }
    }
    int64_t t0 = util::nowMillis();
    std::unique_ptr<Engine> e = createEngine();   // old engine is released first (32-bit address space!)
    {
        std::lock_guard<std::mutex> g(stateMu_);
        engine_.reset(); loadedName_.clear();
    }
    if (!e->load(m.blobPath, opts, err)) return false;
    std::lock_guard<std::mutex> g(stateMu_);
    engine_ = std::move(e);
    loadedName_ = m.name.full(); loadedBlob_ = m.blob; loadedSize_ = m.size; loadedCtx_ = opts.num_ctx;
    busy_ = true;
    loadMs = util::nowMillis() - t0;
    return true;
}

void Runner::touch(int64_t keepAlive) {
    std::lock_guard<std::mutex> g(stateMu_);
    busy_ = false;
    if (keepAlive < 0) expiresAt_ = 0;                          // forever
    else expiresAt_ = util::nowUnix() + keepAlive;
    if (keepAlive == 0) { engine_.reset(); loadedName_.clear(); }
}

std::vector<LoadedInfo> Runner::loaded() {
    std::lock_guard<std::mutex> g(stateMu_);
    std::vector<LoadedInfo> v;
    if (engine_) {
        LoadedInfo i; i.name = loadedName_; i.blob = loadedBlob_; i.size = loadedSize_;
        i.expiresAt = busy_ ? util::nowUnix() + 300 : expiresAt_; i.contextLength = loadedCtx_;
        v.push_back(i);
    }
    return v;
}

void Runner::unload() {
    std::lock_guard<std::mutex> g(stateMu_);
    if (busy_) return;
    engine_.reset(); loadedName_.clear();
}

bool Runner::unloadModel(const std::string& model) {
    ModelName n; std::string e;
    if (!ModelName::parse(model, n, e)) return false;
    std::lock_guard<std::mutex> g(stateMu_);
    if (busy_ || !engine_ || loadedName_ != n.full()) return false;
    engine_.reset(); loadedName_.clear();
    return true;
}

void Runner::reap() {
    std::lock_guard<std::mutex> g(stateMu_);
    if (engine_ && !busy_ && expiresAt_ != 0 && util::nowUnix() >= expiresAt_) { engine_.reset(); loadedName_.clear(); }
}

bool Runner::buildPrompt(const ModelInfo& m, const CompletionRequest& req, Engine* eng, std::string& prompt, std::string& err) {
    if (req.raw && !req.isChat) { prompt = req.prompt; return true; }

    std::vector<Message> msgs;
    std::string system = !req.system.empty() ? req.system : m.system;
    if (req.isChat) {
        msgs = m.messages;
        // A system message in the request replaces the model's default system prompt.
        bool reqHasSystem = false;
        for (size_t i = 0; i < req.messages.size(); i++) if (req.messages[i].role == "system") reqHasSystem = true;
        if (reqHasSystem) system.clear();
        for (size_t i = 0; i < req.messages.size(); i++) msgs.push_back(req.messages[i]);
    } else {
        msgs = m.messages;
        Message u; u.role = "user"; u.content = req.prompt; msgs.push_back(u);
    }

    TemplateInput in; in.system = system; in.messages = msgs;
    std::string tmpl = !req.templ.empty() ? req.templ : m.templ;
    if (!tmpl.empty()) {
        std::string e;
        if (renderTemplate(tmpl, in, prompt, e)) return true;
        fprintf(stderr, "warning: template could not be rendered (%s); using the model's built-in chat template\n", e.c_str());
    }
    // No (usable) TEMPLATE: use the chat template embedded in the GGUF file, else ChatML.
    std::vector<Message> full = msgs;
    if (!system.empty() && (full.empty() || full[0].role != "system")) {
        Message s; s.role = "system"; s.content = system; full.insert(full.begin(), s);
    }
    if (eng && !m.gguf.chatTemplate.empty() && eng->applyChatTemplate(m.gguf.chatTemplate, full, true, prompt)) return true;
    prompt = chatmlPrompt(full, true);
    (void)err;
    return true;
}

bool Runner::complete(const CompletionRequest& req, const std::function<bool(const std::string&)>& onText,
                      RunResult& rr, std::string& err, int& status) {
    status = 500;
    ModelName name;
    if (!ModelName::parse(req.model, name, err)) { status = 400; return false; }
    ModelInfo m;
    if (!store_.get(name, m)) { err = "model '" + req.model + "' not found"; status = 404; return false; }

    std::lock_guard<std::mutex> serial(mu_);
    int64_t t0 = util::nowMillis();
    GenOptions opts = optionsFor(m, req.options);
    int64_t ka = req.keepAliveSec == -2 ? defaultKeepAlive() : req.keepAliveSec;

    int64_t loadMs = 0;
    if (!ensureLoaded(m, opts, ka, loadMs, err)) return false;

    std::string prompt;
    // engine_ is only replaced under mu_, which we hold, so using it without stateMu_ is safe.
    if (!buildPrompt(m, req, engine_.get(), prompt, err)) { touch(ka); return false; }
    bool addSpecial = !(util::startsWith(prompt, "<s>") || util::startsWith(prompt, "<|begin_of_text|>") ||
                        util::startsWith(prompt, "<bos>") || util::startsWith(prompt, "<|startoftext|>"));
    if (req.raw) addSpecial = true;
    if (!getenv("OLLAMA_DEBUG")) {} else fprintf(stderr, "PROMPT:\n%s\n---\n", prompt.c_str());

    StopFilter filter(opts.stop);
    bool cancelled = false;
    bool ok = engine_->generate(prompt, opts, addSpecial,
        [&](const std::string& piece) -> bool {
            std::string out = filter.feed(piece);
            if (!out.empty() && !onText(out)) { cancelled = true; return false; }
            return !filter.stopped();
        }, rr.stats, err);
    if (ok && !cancelled) {
        std::string rest = filter.flush();
        if (!rest.empty()) onText(rest);
    }
    rr.loadMs = loadMs;
    rr.totalMs = util::nowMillis() - t0;
    rr.doneReason = rr.stats.doneReason;
    touch(ka);
    if (!ok) status = 500;
    return ok;
}

bool Runner::embed(const std::string& model, const std::vector<std::string>& inputs, const Json& options,
                   int64_t keepAliveSec, std::vector<std::vector<float> >& out, int& tokens, std::string& err, int& status) {
    status = 500;
    ModelName name;
    if (!ModelName::parse(model, name, err)) { status = 400; return false; }
    ModelInfo m;
    if (!store_.get(name, m)) { err = "model '" + model + "' not found"; status = 404; return false; }
    std::lock_guard<std::mutex> serial(mu_);
    GenOptions opts = optionsFor(m, options);
    int64_t ka = keepAliveSec == -2 ? defaultKeepAlive() : keepAliveSec;
    int64_t loadMs = 0;
    if (!ensureLoaded(m, opts, ka, loadMs, err)) return false;
    bool ok = engine_->embed(inputs, out, tokens, err);
    touch(ka);
    return ok;
}
