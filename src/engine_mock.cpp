// Mock engine: lets the CLI, REST API and model store be exercised without llama.cpp or weights.
#include "engine.h"

#include <cmath>
#include <sstream>

#ifndef XPLLM_HAVE_LLAMA
namespace {
class MockEngine : public Engine {
public:
    MockEngine() : ctx_(2048) {}
    bool load(const std::string& path, const GenOptions& opts, std::string& err) override {
        (void)err; path_ = path; ctx_ = opts.num_ctx; return true;
    }
    bool generate(const std::string& prompt, const GenOptions& opts, bool, const std::function<bool(const std::string&)>& onText,
                  GenStats& st, std::string&) override {
        std::string last = prompt;
        std::string reply = "[mock engine] You said: " + last.substr(last.size() > 200 ? last.size() - 200 : 0);
        std::istringstream in(reply); std::string w; int n = 0;
        st.promptTokens = (int)(prompt.size() / 4) + 1;
        while (in >> w) {
            if (opts.num_predict >= 0 && n >= opts.num_predict) { st.doneReason = "length"; break; }
            if (!onText((n ? " " : "") + w)) break;
            n++;
        }
        st.evalTokens = n; st.evalMs = n; st.promptMs = 1;
        return true;
    }
    bool embed(const std::vector<std::string>& in, std::vector<std::vector<float> >& out, int& tokens, std::string&) override {
        tokens = 0;
        for (size_t i = 0; i < in.size(); i++) {
            std::vector<float> v(8, 0.f);
            for (size_t k = 0; k < in[i].size(); k++) v[k % 8] += (unsigned char)in[i][k] / 255.f;
            float nrm = 0; for (size_t k = 0; k < 8; k++) nrm += v[k] * v[k];
            nrm = std::sqrt(nrm); if (nrm > 0) for (size_t k = 0; k < 8; k++) v[k] /= nrm;
            out.push_back(v); tokens += (int)in[i].size() / 4 + 1;
        }
        return true;
    }
    bool applyChatTemplate(const std::string&, const std::vector<Message>&, bool, std::string&) override { return false; }
    int contextSize() const override { return ctx_; }
    std::string name() const override { return "mock"; }
private:
    std::string path_; int ctx_;
};
}  // namespace

std::unique_ptr<Engine> createEngine() { return std::unique_ptr<Engine>(new MockEngine()); }
bool engineIsMock() { return true; }
#endif
