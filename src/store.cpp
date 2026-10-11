#include "store.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <filesystem>
#include <fstream>

#include "util.h"

using util::joinPath;

// ---------- names ----------
bool ModelName::parse(const std::string& in, ModelName& out, std::string& err) {
    std::string s = util::trim(in);
    // Accept registry prefixes the way Ollama does, then drop them (we only have a local store).
    static const char* prefixes[] = {"registry.ollama.ai/library/", "registry.ollama.ai/", "ollama.com/library/"};
    for (size_t i = 0; i < sizeof prefixes / sizeof *prefixes; i++)
        if (util::startsWith(s, prefixes[i])) { s = s.substr(std::strlen(prefixes[i])); break; }
    if (s.empty()) { err = "model name is empty"; return false; }
    std::string name = s, tag = "latest";
    size_t colon = s.rfind(':');
    if (colon != std::string::npos && s.find('/', colon) == std::string::npos) {
        name = s.substr(0, colon); tag = s.substr(colon + 1);
    }
    if (name.empty() || tag.empty()) { err = "invalid model name: " + in; return false; }
    for (int pass = 0; pass < 2; pass++) {
        const std::string& part = pass ? tag : name;
        for (size_t i = 0; i < part.size(); i++) {
            char c = part[i];
            bool ok = std::isalnum((unsigned char)c) || c == '.' || c == '_' || c == '-' || (pass == 0 && c == '/');
            if (!ok) { err = "invalid character in model name: " + in; return false; }
        }
    }
    if (name.find("..") != std::string::npos || name[0] == '/' || name[0] == '.' ||
        name[name.size() - 1] == '/' || name.find("//") != std::string::npos || tag[0] == '.') {
        err = "invalid model name: " + in; return false;
    }
    out.name = util::lower(name); out.tag = tag;
    return true;
}

// ---------- GGUF header ----------
namespace {
struct Reader {
    std::ifstream f; bool ok = true;
    template <class T> T get() { T v = T(); f.read((char*)&v, sizeof v); if (!f) ok = false; return v; }
    std::string str() {
        uint64_t n = get<uint64_t>();
        if (!ok || n > (1u << 24)) { ok = false; return ""; }
        std::string s((size_t)n, '\0'); f.read(&s[0], (std::streamsize)n); if (!f) ok = false; return s;
    }
};

// Skips/reads one value of the given GGUF type. Stores strings/ints when wanted.
void readValue(Reader& r, uint32_t type, std::string* s, uint64_t* u) {
    static const int sizes[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
    if (type == 8) { std::string v = r.str(); if (s) *s = v; return; }
    if (type == 9) {
        uint32_t et = r.get<uint32_t>(); uint64_t n = r.get<uint64_t>();
        for (uint64_t i = 0; i < n && r.ok; i++) readValue(r, et, 0, 0);
        return;
    }
    if (type > 12) { r.ok = false; return; }
    uint64_t v = 0; r.f.read((char*)&v, sizes[type]); if (!r.f) r.ok = false;  // little-endian host
    if (u) *u = v;
}

const char* fileTypeName(uint32_t t) {
    switch (t) {
        case 0: return "F32"; case 1: return "F16"; case 2: return "Q4_0"; case 3: return "Q4_1";
        case 7: return "Q8_0"; case 8: return "Q5_0"; case 9: return "Q5_1"; case 10: return "Q2_K";
        case 11: return "Q3_K_S"; case 12: return "Q3_K_M"; case 13: return "Q3_K_L"; case 14: return "Q4_K_S";
        case 15: return "Q4_K_M"; case 16: return "Q5_K_S"; case 17: return "Q5_K_M"; case 18: return "Q6_K";
        case 32: return "BF16"; default: return "unknown";
    }
}
}  // namespace

GgufInfo readGgufInfo(const std::string& path) {
    GgufInfo g;
    Reader r;
    r.f.open(std::filesystem::u8path(path), std::ios::binary);
    if (!r.f) return g;
    char magic[4]; r.f.read(magic, 4);
    if (!r.f || std::memcmp(magic, "GGUF", 4) != 0) return g;
    uint32_t version = r.get<uint32_t>();
    if (version < 2 || version > 3) return g;
    g.tensorCount = r.get<uint64_t>();
    uint64_t kv = r.get<uint64_t>();
    for (uint64_t i = 0; i < kv && r.ok; i++) {
        std::string key = r.str(); uint32_t type = r.get<uint32_t>();
        if (!r.ok) break;
        std::string s; uint64_t u = 0;
        readValue(r, type, &s, &u);
        if (key == "general.architecture") g.architecture = s;
        else if (key == "general.name") g.name = s;
        else if (key == "general.size_label") g.sizeLabel = s;
        else if (key == "general.file_type") g.fileType = (uint32_t)u;
        else if (key == "tokenizer.chat_template") g.chatTemplate = s;
        else if (util::endsWith(key, ".context_length")) g.contextLength = u;
        else if (util::endsWith(key, ".embedding_length")) g.embeddingLength = u;
        else if (util::endsWith(key, ".block_count")) g.blockCount = u;
    }
    if (!r.ok) return g;
    g.quantization = fileTypeName(g.fileType);
    g.valid = true;
    return g;
}

// ---------- store ----------
Store::Store(const std::string& root) : root_(root) {}

std::string Store::defaultRoot() {
    std::string m = util::getEnv("OLLAMA_MODELS");
    if (!m.empty()) return m;
    return joinPath(joinPath(util::homeDir(), ".ollama"), "models");
}

std::string Store::blobPath(const std::string& hex) const { return joinPath(root_, "blobs/sha256-" + hex); }
std::string Store::manifestPath(const ModelName& n) const {
    return joinPath(root_, "manifests/" + n.name + "/" + n.tag);
}
std::string Store::newTempPath() const {
    return joinPath(root_, "blobs/partial-" + std::to_string(util::nowMillis()));
}

bool Store::load(const std::string& file, ModelInfo& m) {
    std::string text; Json j; std::string e;
    if (!util::readFile(file, text) || !Json::parse(text, j, &e)) return false;
    m.blob = j.getStr("model");
    m.blobPath = blobPath(m.blob);
    m.size = (uint64_t)j.getNum("size");
    m.modified = (int64_t)j.getNum("modified_at");
    m.system = j.getStr("system"); m.templ = j.getStr("template"); m.license = j.getStr("license");
    if (const Json* p = j.find("params")) m.params = *p;
    m.messages.clear();
    if (const Json* ms = j.find("messages"))
        for (size_t i = 0; i < ms->items().size(); i++) {
            Message x; x.role = ms->items()[i].getStr("role"); x.content = ms->items()[i].getStr("content");
            m.messages.push_back(x);
        }
    m.digest = util::sha256Hex(text).substr(0, 12);
    m.gguf = readGgufInfo(m.blobPath);
    return util::fileExists(m.blobPath);
}

std::vector<ModelInfo> Store::list() {
    std::vector<ModelInfo> out;
    std::string base = joinPath(root_, "manifests");
    std::vector<std::string> files = util::listFiles(base);
    for (size_t i = 0; i < files.size(); i++) {
        std::string rel = files[i].substr(std::min(files[i].size(), base.size() + 1));
        size_t slash = rel.rfind('/');
        if (slash == std::string::npos) continue;
        ModelInfo m; m.name.name = rel.substr(0, slash); m.name.tag = rel.substr(slash + 1);
        if (load(files[i], m)) out.push_back(m);
    }
    std::sort(out.begin(), out.end(), [](const ModelInfo& a, const ModelInfo& b) { return a.modified > b.modified; });
    return out;
}

bool Store::get(const ModelName& n, ModelInfo& out) {
    out = ModelInfo(); out.name = n;
    return load(manifestPath(n), out);
}

bool Store::save(ModelInfo& m, std::string& err) {
    Json j = Json::object();
    j["model"] = m.blob; j["size"] = (long long)m.size; j["modified_at"] = (long long)m.modified;
    j["system"] = m.system; j["template"] = m.templ; j["license"] = m.license;
    j["params"] = m.params;
    Json ms = Json::array();
    for (size_t i = 0; i < m.messages.size(); i++) {
        Json x = Json::object(); x["role"] = m.messages[i].role; x["content"] = m.messages[i].content; ms.push(x);
    }
    j["messages"] = ms;
    std::string text = j.dump();
    if (!util::writeFile(manifestPath(m.name), text)) { err = "cannot write manifest"; return false; }
    m.digest = util::sha256Hex(text).substr(0, 12);
    return true;
}

bool Store::remove(const ModelName& n, std::string& err) {
    std::string f = manifestPath(n);
    if (!util::fileExists(f)) { err = "model '" + n.full() + "' not found"; return false; }
    util::removeFile(f);
    util::removeEmptyDirs(f.substr(0, f.rfind('/')), joinPath(root_, "manifests"));
    pruneBlobs();
    return true;
}

bool Store::copy(const ModelName& from, const ModelName& to, std::string& err) {
    ModelInfo m;
    if (!get(from, m)) { err = "model '" + from.full() + "' not found"; return false; }
    m.name = to; m.modified = util::nowUnix();
    return save(m, err);
}

void Store::pruneBlobs() {
    std::vector<ModelInfo> ms;
    std::string base = joinPath(root_, "manifests");
    std::vector<std::string> files = util::listFiles(base);
    std::vector<std::string> used;
    for (size_t i = 0; i < files.size(); i++) {
        std::string t; Json j;
        if (util::readFile(files[i], t) && Json::parse(t, j)) used.push_back(j.getStr("model"));
    }
    std::vector<std::string> blobs = util::listFiles(joinPath(root_, "blobs"));
    for (size_t i = 0; i < blobs.size(); i++) {
        std::string name = blobs[i].substr(blobs[i].rfind('/') + 1);
        if (!util::startsWith(name, "sha256-")) continue;   // keep partial downloads
        if (std::find(used.begin(), used.end(), name.substr(7)) == used.end()) util::removeFile(blobs[i]);
    }
}

bool Store::importBlob(const std::string& file, bool move, std::string& hex, std::string& err,
                       const std::function<void(const std::string&, uint64_t, uint64_t)>& progress) {
    if (!util::fileExists(file)) { err = "file not found: " + file; return false; }
    hex = util::sha256File(file, [&](uint64_t d, uint64_t t) { if (progress) progress("verifying sha256 digest", d, t); });
    if (hex.empty()) { err = "cannot read " + file; return false; }
    std::string dst = blobPath(hex);
    if (util::fileExists(dst)) { if (move) util::removeFile(file); return true; }
    bool ok = move ? util::moveFile(file, dst) : util::copyFile(file, dst);
    if (!ok) err = "cannot store blob (disk full?)";
    return ok;
}

// ---------- Modelfile ----------
static bool readArg(const std::string& text, size_t& p, std::string& out, std::string& err) {
    while (p < text.size() && (text[p] == ' ' || text[p] == '\t')) p++;
    if (text.compare(p, 3, "\"\"\"") == 0) {
        size_t e = text.find("\"\"\"", p + 3);
        if (e == std::string::npos) { err = "unterminated \"\"\" block"; return false; }
        out = text.substr(p + 3, e - p - 3);
        // A block that starts right after the quotes with a newline keeps Ollama's behaviour: content as written.
        p = e + 3;
        return true;
    }
    size_t e = text.find('\n', p);
    if (e == std::string::npos) e = text.size();
    out = util::trim(text.substr(p, e - p));
    if (out.size() >= 2 && out[0] == '"' && out[out.size() - 1] == '"') out = out.substr(1, out.size() - 2);
    p = e;
    return true;
}

bool Modelfile::parse(const std::string& textIn, Modelfile& mf, std::string& err) {
    std::string text = util::replaceAll(textIn, "\r\n", "\n");
    size_t p = 0;
    while (p < text.size()) {
        while (p < text.size() && std::isspace((unsigned char)text[p])) p++;
        if (p >= text.size()) break;
        if (text[p] == '#') { while (p < text.size() && text[p] != '\n') p++; continue; }
        size_t s = p;
        while (p < text.size() && !std::isspace((unsigned char)text[p])) p++;
        std::string cmd = util::lower(text.substr(s, p - s));
        std::string arg;
        if (!readArg(text, p, arg, err)) return false;
        if (cmd == "from") mf.from = arg;
        else if (cmd == "system") mf.system = arg;
        else if (cmd == "template") mf.templ = arg;
        else if (cmd == "license") mf.license += (mf.license.empty() ? "" : "\n") + arg;
        else if (cmd == "adapter") mf.adapter = arg;
        else if (cmd == "parameter") {
            size_t sp = arg.find_first_of(" \t");
            if (sp == std::string::npos) { err = "PARAMETER needs a name and a value"; return false; }
            std::string v = util::trim(arg.substr(sp + 1));
            if (v.size() >= 2 && v[0] == '"' && v[v.size() - 1] == '"') v = v.substr(1, v.size() - 2);
            mf.params.push_back(std::make_pair(util::lower(arg.substr(0, sp)), v));
        } else if (cmd == "message") {
            size_t sp = arg.find_first_of(" \t");
            if (sp == std::string::npos) { err = "MESSAGE needs a role and content"; return false; }
            Message m; m.role = util::lower(arg.substr(0, sp)); m.content = util::trim(arg.substr(sp + 1));
            if (m.role != "system" && m.role != "user" && m.role != "assistant") { err = "unknown message role: " + m.role; return false; }
            mf.messages.push_back(m);
        } else { err = "unknown instruction: " + cmd; return false; }
    }
    if (mf.from.empty()) { err = "no FROM line"; return false; }
    return true;
}

static std::string quoteBlock(const std::string& s) {
    return s.find('\n') != std::string::npos || s.find('"') != std::string::npos ? "\"\"\"" + s + "\"\"\"" : "\"" + s + "\"";
}

std::string Modelfile::render() const {
    std::string o = "FROM " + from + "\n";
    if (!templ.empty()) o += "TEMPLATE " + quoteBlock(templ) + "\n";
    if (!system.empty()) o += "SYSTEM " + quoteBlock(system) + "\n";
    for (size_t i = 0; i < params.size(); i++) o += "PARAMETER " + params[i].first + " " + params[i].second + "\n";
    for (size_t i = 0; i < messages.size(); i++) o += "MESSAGE " + messages[i].role + " " + messages[i].content + "\n";
    if (!license.empty()) o += "LICENSE " + quoteBlock(license) + "\n";
    return o;
}

Json paramsToJson(const std::vector<std::pair<std::string, std::string> >& p) {
    Json j = Json::object();
    for (size_t i = 0; i < p.size(); i++) {
        const std::string& k = p[i].first; const std::string& v = p[i].second;
        if (k == "stop") {
            if (!j.has("stop")) j["stop"] = Json::array();
            j["stop"].push(Json(v));
            continue;
        }
        char* e = 0; double d = std::strtod(v.c_str(), &e);
        if (e && *e == '\0' && !v.empty()) j[k] = Json(d); else j[k] = Json(v);
    }
    return j;
}

Modelfile modelfileOf(const ModelInfo& m) {
    Modelfile mf;
    mf.from = m.name.full(); mf.system = m.system; mf.templ = m.templ; mf.license = m.license;
    mf.messages = m.messages;
    for (size_t i = 0; i < m.params.members().size(); i++) {
        const std::pair<std::string, Json>& kv = m.params.members()[i];
        if (kv.second.isArray()) {
            for (size_t k = 0; k < kv.second.items().size(); k++)
                mf.params.push_back(std::make_pair(kv.first, kv.second.items()[k].str()));
        } else if (kv.second.isNumber()) {
            Json n = kv.second; mf.params.push_back(std::make_pair(kv.first, n.dump()));
        } else mf.params.push_back(std::make_pair(kv.first, kv.second.str()));
    }
    return mf;
}

bool createModel(Store& store, const ModelName& name, const Modelfile& mf, const std::string& baseDir,
                 std::string& err, const std::function<void(const std::string&)>& status) {
    if (!mf.adapter.empty()) { err = "ADAPTER (LoRA) is not supported in this build"; return false; }
    ModelInfo m; m.name = name;
    ModelName src; std::string e2;
    std::string from = mf.from;
    std::string path = from;
    bool isFile = util::endsWith(util::lower(from), ".gguf") || from.find('/') == 0 || from.find('\\') != std::string::npos ||
                  (from.size() > 1 && from[1] == ':') || util::startsWith(from, "./") || util::startsWith(from, "../");
    ModelInfo base;
    if (!isFile && ModelName::parse(from, src, e2) && store.get(src, base)) {
        m.blob = base.blob; m.size = base.size; m.system = base.system; m.templ = base.templ;
        m.license = base.license; m.params = base.params; m.messages = base.messages;
    } else {
        if (!isFile) { err = "base model '" + from + "' not found; pull/import it first or use FROM <path>.gguf"; return false; }
        if (!(from[0] == '/' || (from.size() > 1 && from[1] == ':') || from[0] == '\\')) path = joinPath(baseDir, from);
        GgufInfo g = readGgufInfo(path);
        if (!g.valid) { err = "not a valid GGUF file: " + path; return false; }
        status("importing model");
        if (!store.importBlob(path, false, m.blob, err, [&](const std::string& s, uint64_t, uint64_t) { (void)s; })) return false;
        m.size = (uint64_t)util::fileSize(store.blobPath(m.blob));
    }
    if (!mf.system.empty()) m.system = mf.system;
    if (!mf.templ.empty()) m.templ = mf.templ;
    if (!mf.license.empty()) m.license = mf.license;
    if (!mf.messages.empty()) m.messages = mf.messages;
    Json np = paramsToJson(mf.params);
    for (size_t i = 0; i < np.members().size(); i++) m.params[np.members()[i].first] = np.members()[i].second;
    m.modified = util::nowUnix();
    status("writing manifest");
    return store.save(m, err);
}
