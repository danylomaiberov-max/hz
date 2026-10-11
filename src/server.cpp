#include "server.h"

#include <cstdio>
#include <memory>

#include "pull.h"
#include "util.h"

namespace {

const char* kJson = "application/json";
const char* kNdjson = "application/x-ndjson";

void sendJson(http::Response& rs, int status, const Json& j) { rs.send(status, kJson, j.dump()); }
void sendError(http::Response& rs, int status, const std::string& msg) {
    Json j = Json::object(); j["error"] = msg; sendJson(rs, status, j);
}

std::string modelOf(const Json& b) {
    std::string m = b.getStr("model");
    return m.empty() ? b.getStr("name") : m;
}

long long ns(int64_t ms) { return (long long)ms * 1000000LL; }

void addStats(Json& j, const RunResult& r) {
    j["done_reason"] = r.doneReason;
    j["total_duration"] = ns(r.totalMs);
    j["load_duration"] = ns(r.loadMs);
    j["prompt_eval_count"] = r.stats.promptTokens;
    j["prompt_eval_duration"] = ns(r.stats.promptMs);
    j["eval_count"] = r.stats.evalTokens;
    j["eval_duration"] = ns(r.stats.evalMs);
}

std::vector<Message> parseMessages(const Json* arr) {
    std::vector<Message> v;
    if (!arr || !arr->isArray()) return v;
    for (size_t i = 0; i < arr->items().size(); i++) {
        const Json& m = arr->items()[i];
        Message x; x.role = m.getStr("role");
        const Json* c = m.find("content");
        if (c && c->isString()) x.content = c->str();
        else if (c && c->isArray()) {   // OpenAI content parts
            for (size_t k = 0; k < c->items().size(); k++) x.content += c->items()[k].getStr("text");
        }
        v.push_back(x);
    }
    return v;
}

Json paramsToMapJson(const Json& p) { return p; }

std::string paramsText(const Json& p) {
    std::string o;
    for (size_t i = 0; i < p.members().size(); i++) {
        const std::pair<std::string, Json>& kv = p.members()[i];
        if (kv.second.isArray()) for (size_t k = 0; k < kv.second.items().size(); k++) o += kv.first + " " + kv.second.items()[k].str() + "\n";
        else o += kv.first + " " + (kv.second.isString() ? kv.second.str() : kv.second.dump()) + "\n";
    }
    return o;
}

}  // namespace

Json modelDetails(const ModelInfo& m) {
    Json d = Json::object();
    d["parent_model"] = ""; d["format"] = "gguf";
    d["family"] = m.gguf.architecture;
    Json fam = Json::array(); if (!m.gguf.architecture.empty()) fam.push(Json(m.gguf.architecture));
    d["families"] = fam;
    d["parameter_size"] = m.gguf.sizeLabel;
    d["quantization_level"] = m.gguf.quantization;
    return d;
}

void ApiServer::handle(const http::Request& rq, http::Response& rs) {
    if (rq.method == "OPTIONS") { rs.send(204, "text/plain", ""); return; }
    const std::string& p = rq.path;
    if (p == "/" || p.empty()) { rs.send(200, "text/plain; charset=utf-8", "Ollama is running"); return; }
    if (rq.method == "HEAD") { rs.send(200, "text/plain", ""); return; }
    if (p == "/api/version") { Json j = Json::object(); j["version"] = OLLAMA_XP_VERSION; sendJson(rs, 200, j); return; }
    if (p == "/api/tags") { tags(rs); return; }
    if (p == "/api/ps") { ps(rs); return; }
    if (p == "/v1/models") { openaiModels(rs); return; }
    if (util::startsWith(p, "/v1/models/")) {
        ModelName n; std::string e; ModelInfo m;
        if (ModelName::parse(http::urlDecode(p.substr(11)), n, e) && store_.get(n, m)) {
            Json j = Json::object(); j["id"] = n.full(); j["object"] = "model"; j["created"] = (long long)m.modified; j["owned_by"] = "library";
            sendJson(rs, 200, j);
        } else sendError(rs, 404, "model not found");
        return;
    }

    if (rq.method != "POST" && rq.method != "DELETE") { sendError(rs, 405, "method not allowed"); return; }
    Json body = Json::object();
    if (!util::trim(rq.body).empty()) {
        std::string e;
        if (!Json::parse(rq.body, body, &e)) { sendError(rs, 400, "invalid JSON: " + e); return; }
    }
    if (p == "/api/generate") generate(body, rs);
    else if (p == "/api/chat") chat(body, rs);
    else if (p == "/api/show") show(body, rs);
    else if (p == "/api/embed") embed(body, false, rs);
    else if (p == "/api/embeddings") embed(body, true, rs);
    else if (p == "/api/pull") pull(body, rs);
    else if (p == "/api/create") create(body, rs);
    else if (p == "/api/copy") copy(body, rs);
    else if (p == "/api/delete") remove(body, rs);
    else if (p == "/v1/chat/completions") openaiChat(body, false, rs);
    else if (p == "/v1/completions") openaiChat(body, true, rs);
    else if (p == "/v1/embeddings") openaiEmbeddings(body, rs);
    else if (p == "/api/push") sendError(rs, 501, "push is not supported in the Windows XP build");
    else if (util::startsWith(p, "/api/blobs/")) sendError(rs, 501, "blob upload is not supported; use FROM <path>.gguf");
    else sendError(rs, 404, "not found");
}

void ApiServer::tags(http::Response& rs) {
    Json arr = Json::array();
    std::vector<ModelInfo> ms = store_.list();
    for (size_t i = 0; i < ms.size(); i++) {
        Json m = Json::object();
        m["name"] = ms[i].name.full(); m["model"] = ms[i].name.full();
        m["modified_at"] = util::iso8601(ms[i].modified);
        m["size"] = (long long)ms[i].size; m["digest"] = ms[i].blob;
        m["details"] = modelDetails(ms[i]);
        arr.push(m);
    }
    Json j = Json::object(); j["models"] = arr; sendJson(rs, 200, j);
}

void ApiServer::ps(http::Response& rs) {
    Json arr = Json::array();
    std::vector<LoadedInfo> l = runner_.loaded();
    for (size_t i = 0; i < l.size(); i++) {
        ModelName n; std::string e; ModelInfo mi;
        Json m = Json::object();
        m["name"] = l[i].name; m["model"] = l[i].name; m["size"] = (long long)l[i].size; m["digest"] = l[i].blob;
        if (ModelName::parse(l[i].name, n, e) && store_.get(n, mi)) m["details"] = modelDetails(mi);
        m["expires_at"] = util::iso8601(l[i].expiresAt ? l[i].expiresAt : util::nowUnix() + 86400LL * 365);
        m["size_vram"] = 0; m["context_length"] = l[i].contextLength;
        arr.push(m);
    }
    Json j = Json::object(); j["models"] = arr; sendJson(rs, 200, j);
}

void ApiServer::show(const Json& body, http::Response& rs) {
    ModelName n; std::string e; ModelInfo m;
    if (!ModelName::parse(modelOf(body), n, e)) { sendError(rs, 400, e); return; }
    if (!store_.get(n, m)) { sendError(rs, 404, "model '" + modelOf(body) + "' not found"); return; }
    Json j = Json::object();
    Modelfile mf = modelfileOf(m); mf.from = m.blobPath;
    j["license"] = m.license;
    j["modelfile"] = "# Modelfile generated by \"ollama show\"\n" + mf.render();
    j["parameters"] = paramsText(m.params);
    j["template"] = m.templ.empty() ? m.gguf.chatTemplate : m.templ;
    j["system"] = m.system;
    j["details"] = modelDetails(m);
    Json msgs = Json::array();
    for (size_t i = 0; i < m.messages.size(); i++) {
        Json x = Json::object(); x["role"] = m.messages[i].role; x["content"] = m.messages[i].content; msgs.push(x);
    }
    j["messages"] = msgs;
    Json info = Json::object();
    info["general.architecture"] = m.gguf.architecture;
    if (!m.gguf.name.empty()) info["general.name"] = m.gguf.name;
    info["general.file_type"] = (long long)m.gguf.fileType;
    if (!m.gguf.architecture.empty()) {
        const std::string& a = m.gguf.architecture;
        info[a + ".context_length"] = (long long)m.gguf.contextLength;
        info[a + ".embedding_length"] = (long long)m.gguf.embeddingLength;
        info[a + ".block_count"] = (long long)m.gguf.blockCount;
    }
    j["model_info"] = info;
    Json caps = Json::array(); caps.push(Json("completion")); j["capabilities"] = caps;
    j["modified_at"] = util::iso8601(m.modified);
    sendJson(rs, 200, j);
}

static CompletionRequest commonRequest(const Json& body) {
    CompletionRequest r;
    r.model = modelOf(body);
    if (const Json* o = body.find("options")) r.options = *o;
    if (const Json* k = body.find("keep_alive")) r.keepAliveSec = Runner::parseKeepAlive(*k, -2);
    return r;
}

void ApiServer::generate(const Json& body, http::Response& rs) {
    CompletionRequest req = commonRequest(body);
    req.isChat = false;
    req.prompt = body.getStr("prompt"); req.system = body.getStr("system"); req.templ = body.getStr("template");
    req.raw = body.getBool("raw");
    bool stream = body.getBool("stream", true);
    if (req.model.empty()) { sendError(rs, 400, "model is required"); return; }

    if (req.prompt.empty() && !req.raw) {   // preload / unload request
        ModelName n; std::string e; ModelInfo m;
        if (!ModelName::parse(req.model, n, e)) { sendError(rs, 400, e); return; }
        if (!store_.get(n, m)) { sendError(rs, 404, "model '" + req.model + "' not found"); return; }
        Json j = Json::object(); j["model"] = req.model; j["created_at"] = util::iso8601(util::nowUnix()); j["response"] = "";
        j["done"] = true;
        if (req.keepAliveSec == 0) { runner_.unloadModel(req.model); j["done_reason"] = "unload"; }
        else {
            CompletionRequest warm = req; warm.prompt = " "; warm.options["num_predict"] = Json(0);
            RunResult rr; std::string err; int st;
            if (!runner_.complete(warm, [](const std::string&) { return true; }, rr, err, st)) { sendError(rs, st, err); return; }
            j["done_reason"] = "load";
        }
        sendJson(rs, 200, j); return;
    }

    std::string created = util::iso8601(util::nowUnix());
    std::string full;
    RunResult rr; std::string err; int status = 500;
    bool started = false, alive = true;
    auto chunkFor = [&](const std::string& text) {
        Json c = Json::object(); c["model"] = req.model; c["created_at"] = created; c["response"] = text; c["done"] = false;
        return c;
    };
    bool ok = runner_.complete(req, [&](const std::string& t) -> bool {
        if (!stream) { full += t; return true; }
        if (!started) { started = true; rs.startStream(200, kNdjson); }
        alive = rs.write(chunkFor(t).dump() + "\n");
        return alive;
    }, rr, err, status);
    if (!ok) {
        if (started) { Json e = Json::object(); e["error"] = err; rs.write(e.dump() + "\n"); }
        else sendError(rs, status, err);
        return;
    }
    Json fin = chunkFor(stream ? "" : full); fin["done"] = true; addStats(fin, rr);
    if (stream) { if (!started) rs.startStream(200, kNdjson); rs.write(fin.dump() + "\n"); }
    else sendJson(rs, 200, fin);
}

void ApiServer::chat(const Json& body, http::Response& rs) {
    CompletionRequest req = commonRequest(body);
    req.isChat = true;
    req.messages = parseMessages(body.find("messages"));
    req.templ = body.getStr("template");
    bool stream = body.getBool("stream", true);
    if (req.model.empty()) { sendError(rs, 400, "model is required"); return; }

    if (req.messages.empty()) {   // load / unload
        ModelName n; std::string e; ModelInfo m;
        if (!ModelName::parse(req.model, n, e)) { sendError(rs, 400, e); return; }
        if (!store_.get(n, m)) { sendError(rs, 404, "model '" + req.model + "' not found"); return; }
        Json j = Json::object(); j["model"] = req.model; j["created_at"] = util::iso8601(util::nowUnix()); j["done"] = true;
        Json msg = Json::object(); msg["role"] = "assistant"; msg["content"] = ""; j["message"] = msg;
        if (req.keepAliveSec == 0) { runner_.unloadModel(req.model); j["done_reason"] = "unload"; }
        else {
            CompletionRequest warm = req; Message u; u.role = "user"; u.content = " "; warm.messages.push_back(u);
            warm.options["num_predict"] = Json(0);
            RunResult rr; std::string err; int st;
            if (!runner_.complete(warm, [](const std::string&) { return true; }, rr, err, st)) { sendError(rs, st, err); return; }
            j["done_reason"] = "load";
        }
        sendJson(rs, 200, j); return;
    }

    std::string created = util::iso8601(util::nowUnix()), full;
    RunResult rr; std::string err; int status = 500; bool started = false;
    auto chunkFor = [&](const std::string& text) {
        Json c = Json::object(); c["model"] = req.model; c["created_at"] = created;
        Json msg = Json::object(); msg["role"] = "assistant"; msg["content"] = text; c["message"] = msg; c["done"] = false;
        return c;
    };
    bool ok = runner_.complete(req, [&](const std::string& t) -> bool {
        if (!stream) { full += t; return true; }
        if (!started) { started = true; rs.startStream(200, kNdjson); }
        return rs.write(chunkFor(t).dump() + "\n");
    }, rr, err, status);
    if (!ok) {
        if (started) { Json e = Json::object(); e["error"] = err; rs.write(e.dump() + "\n"); }
        else sendError(rs, status, err);
        return;
    }
    Json fin = chunkFor(stream ? "" : full); fin["done"] = true; addStats(fin, rr);
    if (stream) { if (!started) rs.startStream(200, kNdjson); rs.write(fin.dump() + "\n"); }
    else sendJson(rs, 200, fin);
}

void ApiServer::embed(const Json& body, bool legacy, http::Response& rs) {
    std::string model = modelOf(body);
    std::vector<std::string> inputs;
    const Json* in = body.find(legacy ? "prompt" : "input");
    if (in && in->isString()) inputs.push_back(in->str());
    else if (in && in->isArray()) for (size_t i = 0; i < in->items().size(); i++) inputs.push_back(in->items()[i].str());
    if (model.empty()) { sendError(rs, 400, "model is required"); return; }
    Json opts = body.find("options") ? *body.find("options") : Json::object();
    int64_t ka = body.find("keep_alive") ? Runner::parseKeepAlive(*body.find("keep_alive"), -2) : -2;
    std::vector<std::vector<float> > out; int tokens = 0; std::string err; int status;
    int64_t t0 = util::nowMillis();
    if (!runner_.embed(model, inputs, opts, ka, out, tokens, err, status)) { sendError(rs, status, err); return; }
    Json j = Json::object();
    auto vec = [](const std::vector<float>& v) { Json a = Json::array(); for (size_t i = 0; i < v.size(); i++) a.push(Json((double)v[i])); return a; };
    if (legacy) { j["embedding"] = out.empty() ? Json::array() : vec(out[0]); }
    else {
        j["model"] = model; Json arr = Json::array();
        for (size_t i = 0; i < out.size(); i++) arr.push(vec(out[i]));
        j["embeddings"] = arr; j["total_duration"] = ns(util::nowMillis() - t0); j["prompt_eval_count"] = tokens;
    }
    sendJson(rs, 200, j);
}

void ApiServer::pull(const Json& body, http::Response& rs) {
    std::string model = modelOf(body);
    ModelName n; std::string err;
    if (!ModelName::parse(model, n, err)) { sendError(rs, 400, err); return; }
    bool stream = body.getBool("stream", true);
    std::string source = body.getStr("source");
    if (source.empty()) source = body.getStr("url");
    bool started = false;
    auto cb = [&](const PullProgress& p) {
        if (!stream) return;
        if (!started) { started = true; rs.startStream(200, kNdjson); }
        Json j = Json::object(); j["status"] = p.status;
        if (!p.digest.empty()) j["digest"] = p.digest;
        if (p.total) { j["total"] = (long long)p.total; j["completed"] = (long long)p.completed; }
        rs.write(j.dump() + "\n");
    };
    bool ok = pullModel(store_, n, source, cb, err);
    if (!ok) {
        if (started) { Json e = Json::object(); e["error"] = err; rs.write(e.dump() + "\n"); }
        else sendError(rs, 500, err);
        return;
    }
    if (!stream) { Json j = Json::object(); j["status"] = "success"; sendJson(rs, 200, j); }
}

void ApiServer::create(const Json& body, http::Response& rs) {
    ModelName n; std::string err;
    if (!ModelName::parse(modelOf(body), n, err)) { sendError(rs, 400, err); return; }
    bool stream = body.getBool("stream", true);
    Modelfile mf;
    std::string text = body.getStr("modelfile");
    if (!text.empty()) { if (!Modelfile::parse(text, mf, err)) { sendError(rs, 400, err); return; } }
    else {
        mf.from = body.getStr("from");
        if (mf.from.empty()) { sendError(rs, 400, "from or modelfile is required"); return; }
    }
    if (body.has("system")) mf.system = body.getStr("system");
    if (body.has("template")) mf.templ = body.getStr("template");
    if (body.has("license")) mf.license = body.getStr("license");
    if (const Json* ps = body.find("parameters"))
        for (size_t i = 0; i < ps->members().size(); i++) {
            const std::pair<std::string, Json>& kv = ps->members()[i];
            if (kv.second.isArray()) for (size_t k = 0; k < kv.second.items().size(); k++) mf.params.push_back(std::make_pair(kv.first, kv.second.items()[k].str()));
            else mf.params.push_back(std::make_pair(kv.first, kv.second.isString() ? kv.second.str() : kv.second.dump()));
        }
    std::vector<Message> ms = parseMessages(body.find("messages"));
    mf.messages.insert(mf.messages.end(), ms.begin(), ms.end());

    bool started = false;
    auto status = [&](const std::string& s) {
        if (!stream) return;
        if (!started) { started = true; rs.startStream(200, kNdjson); }
        Json j = Json::object(); j["status"] = s; rs.write(j.dump() + "\n");
    };
    status("reading model metadata");
    if (!createModel(store_, n, mf, ".", err, status)) {
        if (started) { Json e = Json::object(); e["error"] = err; rs.write(e.dump() + "\n"); }
        else sendError(rs, 400, err);
        return;
    }
    status("success");
    if (!stream) { Json j = Json::object(); j["status"] = "success"; sendJson(rs, 200, j); }
}

void ApiServer::copy(const Json& body, http::Response& rs) {
    ModelName a, b; std::string err;
    if (!ModelName::parse(body.getStr("source"), a, err) || !ModelName::parse(body.getStr("destination"), b, err)) { sendError(rs, 400, err); return; }
    if (!store_.copy(a, b, err)) { sendError(rs, 404, err); return; }
    rs.send(200, kJson, "");
}

void ApiServer::remove(const Json& body, http::Response& rs) {
    ModelName n; std::string err;
    if (!ModelName::parse(modelOf(body), n, err)) { sendError(rs, 400, err); return; }
    runner_.unloadModel(n.full());
    if (!store_.remove(n, err)) { sendError(rs, 404, err); return; }
    rs.send(200, kJson, "");
}

// ---------- OpenAI compatibility ----------
void ApiServer::openaiModels(http::Response& rs) {
    Json arr = Json::array();
    std::vector<ModelInfo> ms = store_.list();
    for (size_t i = 0; i < ms.size(); i++) {
        Json m = Json::object(); m["id"] = ms[i].name.full(); m["object"] = "model";
        m["created"] = (long long)ms[i].modified; m["owned_by"] = "library"; arr.push(m);
    }
    Json j = Json::object(); j["object"] = "list"; j["data"] = arr; sendJson(rs, 200, j);
}

void ApiServer::openaiEmbeddings(const Json& body, http::Response& rs) {
    std::vector<std::string> inputs;
    const Json* in = body.find("input");
    if (in && in->isString()) inputs.push_back(in->str());
    else if (in && in->isArray()) for (size_t i = 0; i < in->items().size(); i++) inputs.push_back(in->items()[i].str());
    std::vector<std::vector<float> > out; int tokens = 0, status; std::string err;
    if (!runner_.embed(body.getStr("model"), inputs, Json::object(), -2, out, tokens, err, status)) { sendError(rs, status, err); return; }
    Json data = Json::array();
    for (size_t i = 0; i < out.size(); i++) {
        Json e = Json::array(); for (size_t k = 0; k < out[i].size(); k++) e.push(Json((double)out[i][k]));
        Json d = Json::object(); d["object"] = "embedding"; d["index"] = (long long)i; d["embedding"] = e; data.push(d);
    }
    Json j = Json::object(); j["object"] = "list"; j["data"] = data; j["model"] = body.getStr("model");
    Json u = Json::object(); u["prompt_tokens"] = tokens; u["total_tokens"] = tokens; j["usage"] = u;
    sendJson(rs, 200, j);
}

void ApiServer::openaiChat(const Json& body, bool completionsApi, http::Response& rs) {
    CompletionRequest req;
    req.model = body.getStr("model");
    req.isChat = !completionsApi;
    if (completionsApi) { const Json* p = body.find("prompt"); req.prompt = p ? (p->isArray() && !p->items().empty() ? p->items()[0].str() : p->str()) : ""; }
    else req.messages = parseMessages(body.find("messages"));
    Json o = Json::object();
    if (body.has("temperature")) o["temperature"] = body.getNum("temperature");
    if (body.has("top_p")) o["top_p"] = body.getNum("top_p");
    if (body.has("seed")) o["seed"] = body.getNum("seed");
    if (body.has("max_tokens")) o["num_predict"] = body.getNum("max_tokens");
    if (body.has("max_completion_tokens")) o["num_predict"] = body.getNum("max_completion_tokens");
    if (body.has("frequency_penalty") && body.getNum("frequency_penalty") > 0) o["repeat_penalty"] = 1.0 + body.getNum("frequency_penalty") * 0.5;
    if (const Json* st = body.find("stop")) o["stop"] = *st;
    req.options = o;
    bool stream = body.getBool("stream");
    std::string id = (completionsApi ? "cmpl-" : "chatcmpl-") + std::to_string(util::nowMillis());
    long long created = (long long)util::nowUnix();

    auto chunk = [&](const std::string& text, const char* finish, bool role) {
        Json c = Json::object(); c["id"] = id; c["created"] = created; c["model"] = req.model;
        Json ch = Json::object(); ch["index"] = 0;
        if (completionsApi) { c["object"] = "text_completion"; ch["text"] = text; }
        else {
            c["object"] = "chat.completion.chunk";
            Json d = Json::object(); if (role) d["role"] = "assistant"; d["content"] = text; ch["delta"] = d;
        }
        ch["finish_reason"] = finish ? Json(finish) : Json();
        Json arr = Json::array(); arr.push(ch); c["choices"] = arr;
        return c;
    };

    std::string full; RunResult rr; std::string err; int status = 500; bool started = false;
    bool ok = runner_.complete(req, [&](const std::string& t) -> bool {
        if (!stream) { full += t; return true; }
        if (!started) { started = true; rs.startStream(200, "text/event-stream"); if (!completionsApi) rs.write("data: " + chunk("", 0, true).dump() + "\n\n"); }
        return rs.write("data: " + chunk(t, 0, false).dump() + "\n\n");
    }, rr, err, status);
    if (!ok) {
        if (started) { Json e = Json::object(); Json m = Json::object(); m["message"] = err; e["error"] = m; rs.write("data: " + e.dump() + "\n\n"); }
        else { Json e = Json::object(); Json m = Json::object(); m["message"] = err; m["type"] = "api_error"; e["error"] = m; sendJson(rs, status, e); }
        return;
    }
    const char* finish = rr.doneReason == "length" ? "length" : "stop";
    if (stream) {
        if (!started) { started = true; rs.startStream(200, "text/event-stream"); }
        rs.write("data: " + chunk("", finish, false).dump() + "\n\ndata: [DONE]\n\n");
        return;
    }
    Json j = Json::object();
    j["id"] = id; j["created"] = created; j["model"] = req.model;
    Json ch = Json::object(); ch["index"] = 0; ch["finish_reason"] = finish;
    if (completionsApi) { j["object"] = "text_completion"; ch["text"] = full; }
    else {
        j["object"] = "chat.completion";
        Json msg = Json::object(); msg["role"] = "assistant"; msg["content"] = full; ch["message"] = msg;
    }
    Json arr = Json::array(); arr.push(ch); j["choices"] = arr;
    Json u = Json::object(); u["prompt_tokens"] = rr.stats.promptTokens; u["completion_tokens"] = rr.stats.evalTokens;
    u["total_tokens"] = rr.stats.promptTokens + rr.stats.evalTokens; j["usage"] = u;
    sendJson(rs, 200, j);
}
