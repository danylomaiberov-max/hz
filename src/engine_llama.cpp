// llama.cpp backend (CPU only, 32-bit friendly).
#ifdef XPLLM_HAVE_LLAMA
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>

#include "engine.h"
#include "llama.h"
#include "util.h"

namespace {

void quietLog(enum ggml_log_level level, const char* text, void*) {
    static const bool debug = !util::getEnv("OLLAMA_DEBUG").empty();
    if (debug || level >= GGML_LOG_LEVEL_ERROR) std::fputs(text, stderr);
}

int autoThreads() {
    unsigned n = std::thread::hardware_concurrency();
    return n ? (int)n : 2;
}

class LlamaEngine : public Engine {
public:
    LlamaEngine() : model_(0), ctx_(0), vocab_(0), nctx_(0) {
        static bool inited = false;
        if (!inited) { llama_log_set(quietLog, 0); llama_backend_init(); inited = true; }
    }
    ~LlamaEngine() override { freeAll(); }

    bool load(const std::string& path, const GenOptions& o, std::string& err) override {
        freeAll();
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        model_ = llama_model_load_from_file(path.c_str(), mp);
        if (!model_) {
            err = "unable to load model (not enough memory or unsupported architecture); "
                  "on 32-bit Windows try a smaller / lower-bit GGUF";
            return false;
        }
        vocab_ = llama_model_get_vocab(model_);
        nctx_ = o.num_ctx;
        threads_ = o.num_thread > 0 ? o.num_thread : autoThreads();
        batch_ = o.num_batch;
        if (!makeContext(false, err)) { freeAll(); return false; }
        return true;
    }

    bool generate(const std::string& prompt, const GenOptions& o, bool addSpecial,
                  const std::function<bool(const std::string&)>& onText, GenStats& st, std::string& err) override {
        if (embedMode_ && !makeContext(false, err)) return false;
        int64_t t0 = util::nowMillis();
        std::vector<llama_token> toks((size_t)prompt.size() + 8);
        int n = llama_tokenize(vocab_, prompt.c_str(), (int)prompt.size(), toks.data(), (int)toks.size(),
                               addSpecial && llama_vocab_get_add_bos(vocab_), true);
        if (n < 0) {
            toks.resize((size_t)-n);
            n = llama_tokenize(vocab_, prompt.c_str(), (int)prompt.size(), toks.data(), (int)toks.size(),
                               addSpecial && llama_vocab_get_add_bos(vocab_), true);
        }
        if (n <= 0) { err = "tokenization failed"; return false; }
        toks.resize((size_t)n);

        int ctxSize = (int)llama_n_ctx(ctx_);
        int room = o.num_predict > 0 ? std::min(o.num_predict, ctxSize / 2) : 0;
        int maxPrompt = std::max(8, ctxSize - 4 - room);
        if ((int)toks.size() > maxPrompt) {   // keep the newest tokens, but always the BOS if present
            std::vector<llama_token> cut;
            if (toks[0] == llama_vocab_bos(vocab_)) cut.push_back(toks[0]);
            size_t keep = (size_t)maxPrompt - cut.size();
            cut.insert(cut.end(), toks.end() - (long)keep, toks.end());
            toks.swap(cut);
        }
        st.promptTokens = (int)toks.size();

        // Reuse the KV cache for the common prefix with the previous request.
        size_t common = 0;
        while (common < toks.size() && common < cache_.size() && toks[common] == cache_[common]) common++;
        if (common == toks.size()) common--;   // always decode at least one token to get logits
        llama_memory_t mem = llama_get_memory(ctx_);
        if (!llama_memory_seq_rm(mem, 0, (llama_pos)common, -1)) { llama_memory_clear(mem, true); common = 0; }
        cache_.resize(common);

        llama_sampler* smpl = makeSampler(o);
        struct Free { llama_sampler* s; ~Free() { llama_sampler_free(s); } } guard = {smpl};

        // Prompt evaluation in chunks.
        size_t pos = common;
        while (pos < toks.size()) {
            int chunk = (int)std::min<size_t>((size_t)o.num_batch, toks.size() - pos);
            llama_batch b = llama_batch_get_one(&toks[pos], chunk);
            if (llama_decode(ctx_, b) != 0) { err = "decode failed (context too large for memory?)"; cache_.clear(); return false; }
            for (int i = 0; i < chunk; i++) cache_.push_back(toks[pos + i]);
            pos += (size_t)chunk;
        }
        st.promptMs = util::nowMillis() - t0;

        int64_t t1 = util::nowMillis();
        std::string held; int produced = 0;
        for (;;) {
            if (o.num_predict >= 0 && produced >= o.num_predict) { st.doneReason = "length"; break; }
            if ((int)cache_.size() >= ctxSize - 1) { st.doneReason = "length"; break; }
            llama_token t = llama_sampler_sample(smpl, ctx_, -1);
            if (llama_vocab_is_eog(vocab_, t)) break;
            llama_sampler_accept(smpl, t);
            char buf[256];
            int len = llama_token_to_piece(vocab_, t, buf, sizeof buf, 0, false);
            if (len < 0) len = 0;
            held.append(buf, (size_t)len);
            produced++;
            llama_batch b = llama_batch_get_one(&t, 1);
            if (llama_decode(ctx_, b) != 0) { err = "decode failed"; return false; }
            cache_.push_back(t);
            size_t ok = utf8CompleteLen(held);
            if (ok) {
                std::string piece = held.substr(0, ok); held.erase(0, ok);
                if (!onText(piece)) { break; }
            }
        }
        if (!held.empty()) onText(held);
        st.evalTokens = produced;
        st.evalMs = util::nowMillis() - t1;
        return true;
    }

    bool embed(const std::vector<std::string>& inputs, std::vector<std::vector<float> >& out, int& tokens,
               std::string& err) override {
        if (!embedMode_ && !makeContext(true, err)) return false;
        tokens = 0;
        int dim = llama_model_n_embd(model_);
        for (size_t i = 0; i < inputs.size(); i++) {
            std::vector<llama_token> toks(inputs[i].size() + 8);
            int n = llama_tokenize(vocab_, inputs[i].c_str(), (int)inputs[i].size(), toks.data(), (int)toks.size(), true, true);
            if (n < 0) { toks.resize((size_t)-n); n = llama_tokenize(vocab_, inputs[i].c_str(), (int)inputs[i].size(), toks.data(), (int)toks.size(), true, true); }
            if (n <= 0) { err = "tokenization failed"; return false; }
            n = std::min(n, (int)llama_n_batch(ctx_));
            llama_memory_clear(llama_get_memory(ctx_), true);
            llama_batch b = llama_batch_get_one(toks.data(), n);
            if (llama_decode(ctx_, b) != 0) { err = "embedding decode failed"; return false; }
            const float* e = llama_get_embeddings_seq(ctx_, 0);
            if (!e) e = llama_get_embeddings_ith(ctx_, n - 1);
            if (!e) { err = "model does not produce embeddings"; return false; }
            std::vector<float> v(e, e + dim);
            double nrm = 0; for (int k = 0; k < dim; k++) nrm += (double)v[(size_t)k] * v[(size_t)k];
            nrm = std::sqrt(nrm);
            if (nrm > 0) for (int k = 0; k < dim; k++) v[(size_t)k] = (float)(v[(size_t)k] / nrm);
            out.push_back(v); tokens += n;
        }
        return true;
    }

    bool applyChatTemplate(const std::string& tmpl, const std::vector<Message>& msgs, bool addAssistant,
                           std::string& out) override {
        std::vector<llama_chat_message> cm(msgs.size());
        for (size_t i = 0; i < msgs.size(); i++) { cm[i].role = msgs[i].role.c_str(); cm[i].content = msgs[i].content.c_str(); }
        std::vector<char> buf(4096);
        int n = llama_chat_apply_template(tmpl.c_str(), cm.data(), cm.size(), addAssistant, buf.data(), (int)buf.size());
        if (n > (int)buf.size()) {
            buf.resize((size_t)n);
            n = llama_chat_apply_template(tmpl.c_str(), cm.data(), cm.size(), addAssistant, buf.data(), (int)buf.size());
        }
        if (n < 0) return false;
        out.assign(buf.data(), (size_t)n);
        return true;
    }

    int contextSize() const override { return ctx_ ? (int)llama_n_ctx(ctx_) : nctx_; }
    std::string name() const override { return "llama.cpp"; }

private:
    bool makeContext(bool embeddings, std::string& err) {
        if (ctx_) { llama_free(ctx_); ctx_ = 0; }
        cache_.clear();
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = (uint32_t)nctx_;
        cp.n_batch = (uint32_t)batch_;
        cp.n_ubatch = (uint32_t)std::min(batch_, 128);
        cp.n_threads = threads_;
        cp.n_threads_batch = threads_;
        cp.embeddings = embeddings;
        if (embeddings) cp.pooling_type = LLAMA_POOLING_TYPE_UNSPECIFIED;
        ctx_ = llama_init_from_model(model_, cp);
        if (!ctx_) { err = "unable to create context (reduce num_ctx; 32-bit address space is limited)"; return false; }
        embedMode_ = embeddings;
        return true;
    }

    llama_sampler* makeSampler(const GenOptions& o) {
        llama_sampler* s = llama_sampler_chain_init(llama_sampler_chain_default_params());
        if (o.repeat_penalty != 1.0f && o.repeat_last_n != 0)
            llama_sampler_chain_add(s, llama_sampler_init_penalties(llama_vocab_n_tokens(vocab_),
                                                                    o.repeat_last_n < 0 ? (int)llama_n_ctx(ctx_) : o.repeat_last_n,
                                                                    o.repeat_penalty, 0.0f, 0.0f));
        if (o.temperature <= 0.0f) {
            llama_sampler_chain_add(s, llama_sampler_init_greedy());
        } else {
            if (o.top_k > 0) llama_sampler_chain_add(s, llama_sampler_init_top_k(o.top_k));
            if (o.top_p < 1.0f) llama_sampler_chain_add(s, llama_sampler_init_top_p(o.top_p, 1));
            if (o.min_p > 0.0f) llama_sampler_chain_add(s, llama_sampler_init_min_p(o.min_p, 1));
            llama_sampler_chain_add(s, llama_sampler_init_temp(o.temperature));
            uint32_t seed = o.seed >= 0 ? (uint32_t)o.seed : (uint32_t)util::nowMillis();
            llama_sampler_chain_add(s, llama_sampler_init_dist(seed));
        }
        return s;
    }

    void freeAll() {
        if (ctx_) { llama_free(ctx_); ctx_ = 0; }
        if (model_) { llama_model_free(model_); model_ = 0; }
        cache_.clear();
    }

    llama_model* model_; llama_context* ctx_; const llama_vocab* vocab_;
    int nctx_, threads_ = 2, batch_ = 256; bool embedMode_ = false;
    std::vector<llama_token> cache_;
};

}  // namespace

std::unique_ptr<Engine> createEngine() { return std::unique_ptr<Engine>(new LlamaEngine()); }
bool engineIsMock() { return false; }
#endif
