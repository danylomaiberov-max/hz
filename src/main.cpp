// ollama-xp: Ollama-style local LLM runner for 32-bit Windows XP (also builds on Linux for testing).
#include <algorithm>
#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iterator>
#include <thread>

#include "http.h"
#include "json.h"
#include "pull.h"
#include "runner.h"
#include "server.h"
#include "store.h"
#include "util.h"

using util::consoleWrite;

static std::atomic<bool> g_cancel(false);
static std::atomic<bool> g_generating(false);

static void onSigint(int) {
    if (g_generating) g_cancel = true;
    else { std::fputs("\n", stderr); std::_Exit(130); }
    std::signal(SIGINT, onSigint);
}

static void err(const std::string& m) { std::fprintf(stderr, "Error: %s\n", m.c_str()); }

static const char* kUsage =
    "Large language model runner (Windows XP / 32-bit edition)\n\n"
    "Usage:\n"
    "  ollama [flags]\n"
    "  ollama [command]\n\n"
    "Available Commands:\n"
    "  serve       Start ollama\n"
    "  create      Create a model from a Modelfile\n"
    "  show        Show information for a model\n"
    "  run         Run a model\n"
    "  stop        Stop a running model\n"
    "  pull        Pull a model from an http:// URL or import a local .gguf\n"
    "  list        List models\n"
    "  ps          List running models\n"
    "  cp          Copy a model\n"
    "  rm          Remove a model\n"
    "  help        Help about any command\n\n"
    "Flags:\n"
    "  -h, --help      help for ollama\n"
    "  -v, --version   Show version information\n\n"
    "Use \"ollama [command] --help\" for more information about a command.\n";

// ---------- helpers ----------
static std::string hostSpec() { return util::getEnv("OLLAMA_HOST", "127.0.0.1:11434"); }

static bool apiCall(const std::string& method, const std::string& path, const std::string& body, Json& out, std::string& e) {
    http::ClientOptions o; o.method = method; o.body = body; o.timeoutSec = 600;
    std::string resp; http::ClientResult r;
    if (!http::request(http::hostToUrl(hostSpec()) + path, o, nullptr,
            [&](const char* p, size_t n) { resp.append(p, n); return true; }, r, e)) {
        e = "could not connect to ollama server (" + hostSpec() + "); start it with `ollama serve`";
        return false;
    }
    if (!resp.empty() && !Json::parse(resp, out)) out = Json::object();
    if (r.status >= 400) { e = out.getStr("error", "HTTP " + std::to_string(r.status)); return false; }
    return true;
}

static void printTable(const std::vector<std::vector<std::string> >& rows) {
    std::vector<size_t> w;
    for (size_t r = 0; r < rows.size(); r++)
        for (size_t c = 0; c < rows[r].size(); c++) {
            if (w.size() <= c) w.push_back(0);
            w[c] = std::max(w[c], rows[r][c].size());
        }
    for (size_t r = 0; r < rows.size(); r++) {
        std::string line;
        for (size_t c = 0; c < rows[r].size(); c++) {
            line += rows[r][c];
            if (c + 1 < rows[r].size()) line += std::string(w[c] - rows[r][c].size() + 4, ' ');
        }
        consoleWrite(line + "\n");
    }
}

static bool parseName(const std::string& s, ModelName& n) {
    std::string e;
    if (!ModelName::parse(s, n, e)) { err(e); return false; }
    return true;
}

static std::string rate(int count, int64_t ms) {
    char b[64];
    std::snprintf(b, sizeof b, "%.2f tokens/s", ms > 0 ? count * 1000.0 / (double)ms : 0.0);
    return b;
}
static std::string dur(int64_t ms) {
    char b[64];
    if (ms < 1000) std::snprintf(b, sizeof b, "%lldms", (long long)ms);
    else std::snprintf(b, sizeof b, "%.3fs", ms / 1000.0);
    return b;
}

static void printStats(const RunResult& r) {
    std::string s;
    s += "\ntotal duration:       " + dur(r.totalMs) + "\n";
    s += "load duration:        " + dur(r.loadMs) + "\n";
    s += "prompt eval count:    " + std::to_string(r.stats.promptTokens) + " token(s)\n";
    s += "prompt eval duration: " + dur(r.stats.promptMs) + "\n";
    s += "prompt eval rate:     " + rate(r.stats.promptTokens, r.stats.promptMs) + "\n";
    s += "eval count:           " + std::to_string(r.stats.evalTokens) + " token(s)\n";
    s += "eval duration:        " + dur(r.stats.evalMs) + "\n";
    s += "eval rate:            " + rate(r.stats.evalTokens, r.stats.evalMs) + "\n";
    std::fputs(s.c_str(), stderr);
}

static void pullProgress(const PullProgress& p) {
    static std::string lastStatus;
    if (p.total == 0) {
        if (p.status != lastStatus) { lastStatus = p.status; std::fprintf(stderr, "\r%-78s\n", p.status.c_str()); }
        return;
    }
    static int64_t t0 = util::nowMillis(); static uint64_t c0 = 0;
    int pct = (int)(p.completed * 100 / p.total);
    int bars = pct * 20 / 100;
    int64_t dt = util::nowMillis() - t0;
    if (p.completed < c0 || lastStatus != p.status) { t0 = util::nowMillis(); c0 = p.completed; dt = 0; }
    lastStatus = p.status;
    std::string speed = dt > 500 ? util::humanBytes((uint64_t)((p.completed - c0) * 1000.0 / (double)dt)) + "/s" : "";
    std::fprintf(stderr, "\r%s %3d%% [%s%s] %s/%s %s   ", p.status.c_str(), pct, std::string((size_t)bars, '#').c_str(),
                 std::string((size_t)(20 - bars), '.').c_str(), util::humanBytes(p.completed).c_str(),
                 util::humanBytes(p.total).c_str(), speed.c_str());
    if (p.completed >= p.total) { std::fputs("\n", stderr); lastStatus.clear(); }
}

// ---------- commands ----------
static int cmdList(Store& store) {
    std::vector<ModelInfo> ms = store.list();
    std::vector<std::vector<std::string> > rows;
    std::vector<std::string> head; head.push_back("NAME"); head.push_back("ID"); head.push_back("SIZE"); head.push_back("MODIFIED");
    rows.push_back(head);
    for (size_t i = 0; i < ms.size(); i++) {
        std::vector<std::string> r;
        r.push_back(ms[i].name.full()); r.push_back(ms[i].blob.substr(0, 12));
        r.push_back(util::humanBytes(ms[i].size)); r.push_back(util::humanAgo(ms[i].modified));
        rows.push_back(r);
    }
    printTable(rows);
    return 0;
}

static int cmdPs() {
    Json j; std::string e;
    if (!apiCall("GET", "/api/ps", "", j, e)) { err(e); return 1; }
    std::vector<std::vector<std::string> > rows;
    std::vector<std::string> head; head.push_back("NAME"); head.push_back("ID"); head.push_back("SIZE");
    head.push_back("PROCESSOR"); head.push_back("CONTEXT"); head.push_back("UNTIL");
    rows.push_back(head);
    const Json* ms = j.find("models");
    if (ms) for (size_t i = 0; i < ms->items().size(); i++) {
        const Json& m = ms->items()[i];
        std::vector<std::string> r;
        r.push_back(m.getStr("name")); r.push_back(m.getStr("digest").substr(0, 12));
        r.push_back(util::humanBytes((uint64_t)m.getNum("size"))); r.push_back("100% CPU");
        r.push_back(std::to_string((long long)m.getNum("context_length")));
        int64_t exp = util::parseIso8601(m.getStr("expires_at"));
        r.push_back(exp > util::nowUnix() + 86400LL * 300 ? "Forever" : util::humanFromNow(exp));
        rows.push_back(r);
    }
    printTable(rows);
    return 0;
}

static int cmdStop(const std::vector<std::string>& a) {
    if (a.size() != 1) { err("usage: ollama stop MODEL"); return 1; }
    Json req = Json::object(); req["model"] = a[0]; req["keep_alive"] = 0; Json out; std::string e;
    if (!apiCall("POST", "/api/generate", req.dump(), out, e)) { err(e); return 1; }
    return 0;
}

static int cmdShow(Store& store, std::vector<std::string> a) {
    bool fModelfile = false, fParams = false, fTemplate = false, fSystem = false, fLicense = false;
    std::string name;
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i] == "--modelfile") fModelfile = true; else if (a[i] == "--parameters") fParams = true;
        else if (a[i] == "--template") fTemplate = true; else if (a[i] == "--system") fSystem = true;
        else if (a[i] == "--license") fLicense = true; else name = a[i];
    }
    ModelName n; ModelInfo m;
    if (name.empty()) { err("usage: ollama show MODEL [flags]"); return 1; }
    if (!parseName(name, n)) return 1;
    if (!store.get(n, m)) { err("model '" + name + "' not found"); return 1; }
    if (fModelfile) { Modelfile mf = modelfileOf(m); mf.from = m.blobPath; consoleWrite("# Modelfile generated by \"ollama show\"\n" + mf.render()); return 0; }
    if (fParams) { Modelfile mf = modelfileOf(m); std::string o; for (size_t i = 0; i < mf.params.size(); i++) o += mf.params[i].first + " " + mf.params[i].second + "\n"; consoleWrite(o); return 0; }
    if (fTemplate) { consoleWrite((m.templ.empty() ? m.gguf.chatTemplate : m.templ) + "\n"); return 0; }
    if (fSystem) { consoleWrite(m.system + "\n"); return 0; }
    if (fLicense) { consoleWrite(m.license + "\n"); return 0; }

    std::vector<std::vector<std::string> > rows;
    #define ROW(k, v) do { std::vector<std::string> r; r.push_back(std::string("    ") + k); r.push_back(v); rows.push_back(r); } while (0)
    consoleWrite("  Model\n");
    ROW("architecture", m.gguf.architecture);
    ROW("parameters", m.gguf.sizeLabel);
    ROW("context length", std::to_string((unsigned long long)m.gguf.contextLength));
    ROW("embedding length", std::to_string((unsigned long long)m.gguf.embeddingLength));
    ROW("quantization", m.gguf.quantization);
    printTable(rows);
    if (!m.params.members().empty()) {
        consoleWrite("\n  Parameters\n"); rows.clear();
        Modelfile mf = modelfileOf(m);
        for (size_t i = 0; i < mf.params.size(); i++) {
            std::vector<std::string> r; r.push_back("    " + mf.params[i].first); r.push_back(mf.params[i].second); rows.push_back(r);
        }
        printTable(rows);
    }
    if (!m.system.empty()) consoleWrite("\n  System\n    " + util::replaceAll(m.system, "\n", "\n    ") + "\n");
    if (!m.license.empty()) consoleWrite("\n  License\n    " + util::replaceAll(m.license.substr(0, 200), "\n", "\n    ") + "\n");
    #undef ROW
    return 0;
}

static int cmdRm(Store& store, const std::vector<std::string>& a) {
    if (a.empty()) { err("usage: ollama rm MODEL [MODEL...]"); return 1; }
    int rc = 0;
    for (size_t i = 0; i < a.size(); i++) {
        ModelName n; std::string e;
        if (!parseName(a[i], n)) { rc = 1; continue; }
        if (!store.remove(n, e)) { err(e); rc = 1; continue; }
        consoleWrite("deleted '" + a[i] + "'\n");
    }
    return rc;
}

static int cmdCp(Store& store, const std::vector<std::string>& a) {
    if (a.size() != 2) { err("usage: ollama cp SOURCE DESTINATION"); return 1; }
    ModelName s, d; std::string e;
    if (!parseName(a[0], s) || !parseName(a[1], d)) return 1;
    if (!store.copy(s, d, e)) { err(e); return 1; }
    consoleWrite("copied '" + a[0] + "' to '" + a[1] + "'\n");
    return 0;
}

static int cmdPull(Store& store, const std::vector<std::string>& a) {
    std::vector<std::string> pos;
    for (size_t i = 0; i < a.size(); i++) if (a[i] != "--insecure") pos.push_back(a[i]);
    if (pos.empty() || pos.size() > 2) { err("usage: ollama pull MODEL [URL|FILE.gguf]"); return 1; }
    ModelName n; if (!parseName(pos[0], n)) return 1;
    std::string e;
    if (!pullModel(store, n, pos.size() > 1 ? pos[1] : "", pullProgress, e)) { err(e); return 1; }
    return 0;
}

static int cmdCreate(Store& store, const std::vector<std::string>& a) {
    std::string name, file = "Modelfile";
    for (size_t i = 0; i < a.size(); i++) {
        if ((a[i] == "-f" || a[i] == "--file") && i + 1 < a.size()) file = a[++i];
        else if (a[i] == "-q" || a[i] == "--quantize") { err("quantization is not supported in this build"); return 1; }
        else name = a[i];
    }
    if (name.empty()) { err("usage: ollama create MODEL [-f Modelfile]"); return 1; }
    ModelName n; if (!parseName(name, n)) return 1;
    std::string text, e;
    if (!util::readFile(file, text)) { err("cannot read " + file); return 1; }
    Modelfile mf;
    if (!Modelfile::parse(text, mf, e)) { err(e); return 1; }
    std::string base = file.find_last_of("/\\") == std::string::npos ? "." : file.substr(0, file.find_last_of("/\\"));
    if (!createModel(store, n, mf, base, e, [](const std::string& s) { consoleWrite(s + "\n"); })) { err(e); return 1; }
    consoleWrite("success\n");
    return 0;
}

static int cmdServe(Store& store) {
    http::startup();
    std::string spec = hostSpec();
    std::string url = http::hostToUrl(spec);                    // http://host:port
    std::string hp = url.substr(7);
    size_t colon = hp.rfind(':');
    std::string host = hp.substr(0, colon); int port = std::atoi(hp.c_str() + colon + 1);
    if (host.empty()) host = "127.0.0.1";

    Runner runner(store); ApiServer api(store, runner);
    http::Server srv; std::string e;
    if (!srv.listen(host, port, e)) { err(e); return 1; }
    std::fprintf(stderr, "ollama-xp %s listening on %s:%d (engine: %s, models: %s)\n", OLLAMA_XP_VERSION, host.c_str(), port,
                 engineIsMock() ? "MOCK - no llama.cpp compiled in" : "llama.cpp", store.root().c_str());
    std::thread([&runner] { for (;;) { std::this_thread::sleep_for(std::chrono::seconds(5)); runner.reap(); } }).detach();
    srv.serve([&](const http::Request& rq, http::Response& rs) { api.handle(rq, rs); });
    return 0;
}

// ---------- run / REPL ----------
struct ChatSession {
    std::string model, system;
    Json options = Json::object();
    std::vector<Message> history;
    bool verbose = false, jsonFormat = false;
    int64_t keepAlive = -2;
};

static bool chatTurn(Runner& runner, ChatSession& s, const std::string& userText) {
    CompletionRequest req; req.isChat = true; req.model = s.model; req.options = s.options; req.keepAliveSec = s.keepAlive;
    req.messages = s.history;
    if (!s.system.empty()) { Message m; m.role = "system"; m.content = s.system; req.messages.insert(req.messages.begin(), m); }
    Message u; u.role = "user"; u.content = userText; req.messages.push_back(u);
    std::string reply, e; RunResult rr; int status = 0;
    g_cancel = false; g_generating = true;
    bool ok = runner.complete(req, [&](const std::string& t) { reply += t; consoleWrite(t); return !g_cancel; }, rr, e, status);
    g_generating = false;
    consoleWrite("\n");
    if (!ok) { err(e); return false; }
    if (g_cancel) { g_cancel = false; return true; }   // interrupted: drop the partial turn like Ollama
    s.history.push_back(u);
    Message a; a.role = "assistant"; a.content = reply; s.history.push_back(a);
    if (s.verbose) printStats(rr);
    return true;
}

static void replHelp() {
    consoleWrite(
        "Available Commands:\n"
        "  /set            Set session variables\n"
        "  /show           Show model information\n"
        "  /load <model>   Load a session or model\n"
        "  /save <model>   Save your current session\n"
        "  /clear          Clear session context\n"
        "  /bye            Exit\n"
        "  /?, /help       Help for a command\n"
        "  /? shortcuts    Help for keyboard shortcuts\n\n"
        "Use \"\"\" to begin a multi-line message.\n\n");
}

static int repl(Store& store, Runner& runner, ChatSession& s) {
    bool tty = util::stdinIsTty();
    if (tty) consoleWrite(">>> Send a message (/? for help)\n");
    std::string line;
    for (;;) {
        if (!util::readLine(tty ? ">>> " : "", line)) break;
        std::string t = util::trim(line);
        if (t.empty()) continue;
        if (util::startsWith(t, "\"\"\"")) {   // multi-line input
            std::string text = t.substr(3), more;
            bool closed = false;
            if (text.size() >= 3 && text.compare(text.size() - 3, 3, "\"\"\"") == 0) { text.erase(text.size() - 3); closed = true; }
            while (!closed && util::readLine(tty ? "... " : "", more)) {
                size_t q = more.find("\"\"\"");
                if (q != std::string::npos) { text += "\n" + more.substr(0, q); closed = true; }
                else text += "\n" + more;
            }
            t = util::trim(text);
            if (t.empty()) continue;
        } else if (t[0] == '/') {
            std::vector<std::string> w;
            { size_t p = 0; while (p < t.size()) { while (p < t.size() && t[p] == ' ') p++; size_t b = p; while (p < t.size() && t[p] != ' ') p++; if (p > b) w.push_back(t.substr(b, p - b)); } }
            const std::string& c = w[0];
            if (c == "/bye" || c == "/exit") break;
            else if (c == "/?" || c == "/help") replHelp();
            else if (c == "/clear") { s.history.clear(); consoleWrite("Cleared session context\n"); }
            else if (c == "/set" && w.size() >= 2) {
                if (w[1] == "verbose") { s.verbose = true; consoleWrite("Set 'verbose' mode.\n"); }
                else if (w[1] == "quiet") { s.verbose = false; consoleWrite("Set 'quiet' mode.\n"); }
                else if (w[1] == "nohistory" || w[1] == "history" || w[1] == "wordwrap" || w[1] == "nowordwrap") consoleWrite("Set '" + w[1] + "' mode.\n");
                else if (w[1] == "format" && w.size() > 2) { s.jsonFormat = w[2] == "json"; consoleWrite("Set format to '" + w[2] + "' mode.\n"); }
                else if (w[1] == "noformat") { s.jsonFormat = false; consoleWrite("Disabled format.\n"); }
                else if (w[1] == "system") {
                    size_t p = t.find("system") + 6; s.system = util::trim(t.substr(p));
                    if (s.system.size() >= 6 && s.system.compare(0, 3, "\"\"\"") == 0) s.system = s.system.substr(3, s.system.size() - 6);
                    consoleWrite("Set system message.\n");
                } else if (w[1] == "parameter" && w.size() >= 4) {
                    Json v; std::string val = t.substr(t.find(w[2], t.find("parameter") + 9) + w[2].size());
                    val = util::trim(val);
                    char* endp = 0; double d = std::strtod(val.c_str(), &endp);
                    if (w[2] == "stop") { if (!s.options.has("stop")) s.options["stop"] = Json::array(); s.options["stop"].push(Json(val)); }
                    else if (endp && *endp == 0) s.options[w[2]] = Json(d); else s.options[w[2]] = Json(val);
                    consoleWrite("Set parameter '" + w[2] + "' to '" + val + "'\n");
                } else consoleWrite("Unknown command '" + t + "'. Type /? for help\n");
            }
            else if (c == "/show") {
                std::vector<std::string> a; a.push_back(s.model);
                std::string what = w.size() > 1 ? w[1] : "info";
                if (what == "modelfile") a.push_back("--modelfile"); else if (what == "parameters") a.push_back("--parameters");
                else if (what == "system") { consoleWrite((s.system.empty() ? "No system message was specified for this model.\n" : s.system + "\n")); continue; }
                else if (what == "template") a.push_back("--template"); else if (what == "license") a.push_back("--license");
                cmdShow(store, a);
            }
            else if (c == "/save" && w.size() == 2) {
                ModelName src, dst; std::string e; ModelInfo m;
                if (parseName(s.model, src) && parseName(w[1], dst) && store.get(src, m)) {
                    m.name = dst; m.system = s.system.empty() ? m.system : s.system;
                    for (size_t i = 0; i < s.history.size(); i++) m.messages.push_back(s.history[i]);
                    Json np = s.options; for (size_t i = 0; i < np.members().size(); i++) m.params[np.members()[i].first] = np.members()[i].second;
                    m.modified = util::nowUnix();
                    if (store.save(m, e)) consoleWrite("Created new model '" + w[1] + "'\n"); else err(e);
                }
            }
            else if (c == "/load" && w.size() == 2) {
                ModelName n; ModelInfo m;
                if (parseName(w[1], n) && store.get(n, m)) { s.model = n.full(); s.history.clear(); consoleWrite("Loading model '" + w[1] + "'\n"); }
                else err("model '" + w[1] + "' not found");
            }
            else consoleWrite("Unknown command '" + c + "'. Type /? for help\n");
            continue;
        }
        if (s.jsonFormat) t += "\n\nRespond only with valid JSON.";
        chatTurn(runner, s, t);
    }
    return 0;
}

static int cmdRun(Store& store, const std::vector<std::string>& a) {
    ChatSession s; std::string name; std::vector<std::string> words;
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i] == "--verbose") s.verbose = true;
        else if (a[i] == "--format" && i + 1 < a.size()) s.jsonFormat = a[++i] == "json";
        else if (a[i] == "--keepalive" && i + 1 < a.size()) s.keepAlive = Runner::parseKeepAlive(Json(a[++i]), -2);
        else if (a[i] == "--nowordwrap" || a[i] == "--insecure" || a[i] == "--hidethinking") {}
        else if (name.empty()) name = a[i];
        else words.push_back(a[i]);
    }
    if (name.empty()) { err("usage: ollama run MODEL [PROMPT]"); return 1; }
    ModelName n; if (!parseName(name, n)) return 1;
    ModelInfo mi;
    if (!store.get(n, mi)) {
        std::string e;
        std::string src = resolveSource(store, n);
        if (src.empty()) { err("model '" + name + "' not found. Import it with `ollama pull " + n.full() + " <url|file.gguf>` or `ollama create`"); return 1; }
        if (!pullModel(store, n, src, pullProgress, e)) { err(e); return 1; }
        store.get(n, mi);
    }
    s.model = n.full();
    std::string prompt;
    for (size_t i = 0; i < words.size(); i++) prompt += (i ? " " : "") + words[i];
    if (!util::stdinIsTty()) {   // piped input is prepended to the prompt
        std::string in((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
        prompt = util::trim(in + (prompt.empty() ? "" : "\n\n" + prompt));
    }
    if (s.jsonFormat && !prompt.empty()) prompt += "\n\nRespond only with valid JSON.";

    Runner runner(store);
    std::signal(SIGINT, onSigint);
    if (!prompt.empty()) return chatTurn(runner, s, prompt) ? 0 : 1;
    return repl(store, runner, s);
}

int main(int argc, char** argv) {
    util::setupConsoleUtf8();
    std::vector<std::string> a(argv + 1, argv + argc);
    if (a.empty() || a[0] == "-h" || a[0] == "--help" || a[0] == "help") { consoleWrite(kUsage); return 0; }
    if (a[0] == "-v" || a[0] == "--version" || a[0] == "version") {
        consoleWrite(std::string("ollama version is ") + OLLAMA_XP_VERSION + (engineIsMock() ? " (mock engine)" : " (llama.cpp)") + "\n");
        return 0;
    }
    std::string cmd = a[0]; a.erase(a.begin());
    Store store(Store::defaultRoot());
    if (cmd == "serve" || cmd == "start") return cmdServe(store);
    if (cmd == "run") return cmdRun(store, a);
    if (cmd == "list" || cmd == "ls") return cmdList(store);
    if (cmd == "ps") return cmdPs();
    if (cmd == "stop") return cmdStop(a);
    if (cmd == "show") return cmdShow(store, a);
    if (cmd == "rm") return cmdRm(store, a);
    if (cmd == "cp") return cmdCp(store, a);
    if (cmd == "pull") return cmdPull(store, a);
    if (cmd == "create") return cmdCreate(store, a);
    if (cmd == "push") { err("push is not supported in the Windows XP build"); return 1; }
    std::fprintf(stderr, "Error: unknown command \"%s\" for \"ollama\"\n", cmd.c_str());
    return 1;
}
