#include "pull.h"

#include <cctype>
#include <filesystem>
#include <fstream>

#include "http.h"
#include "util.h"

std::string resolveSource(const Store& store, const ModelName& name) {
    std::string cat;
    std::string path = util::joinPath(store.root(), "../catalog.txt");
    if (util::readFile(path, cat)) {
        std::vector<std::string> lines = util::split(util::replaceAll(cat, "\r\n", "\n"), '\n');
        for (size_t i = 0; i < lines.size(); i++) {
            std::string l = util::trim(lines[i]);
            if (l.empty() || l[0] == '#') continue;
            size_t sp = l.find_first_of(" \t");
            if (sp == std::string::npos) continue;
            ModelName n; std::string e;
            if (ModelName::parse(l.substr(0, sp), n, e) && n.full() == name.full()) return util::trim(l.substr(sp + 1));
        }
    }
    std::string mirror = util::getEnv("OLLAMA_MIRROR");
    if (!mirror.empty()) {
        std::string m = util::replaceAll(mirror, "{name}", util::replaceAll(name.name, "/", "-"));
        return util::replaceAll(m, "{tag}", name.tag);
    }
    return "";
}

static bool download(Store& store, const std::string& url, std::string& file, const PullCallback& cb, std::string& err) {
    // Resume support: partial file name derives from the URL.
    file = util::joinPath(store.root(), "blobs/partial-" + util::sha256Hex(url).substr(0, 16));
    util::makeDirs(util::joinPath(store.root(), "blobs"));
    uint64_t have = util::fileExists(file) ? (uint64_t)util::fileSize(file) : 0;

    http::ClientOptions opt; opt.rangeStart = have; opt.timeoutSec = 120;
    std::ofstream out;
    uint64_t total = 0, done = 0; bool ok = true; std::string werr;
    http::ClientResult r;
    int64_t lastTick = 0;
    bool net = http::request(url, opt,
        [&](const http::ClientResult& h) -> bool {
            if (h.status == 416) { done = have; total = have; return false; }   // already complete
            if (h.status != 200 && h.status != 206) { werr = "server returned HTTP " + std::to_string(h.status); ok = false; return false; }
            bool resumed = h.status == 206 && have > 0;
            out.open(std::filesystem::u8path(file), resumed ? std::ios::binary | std::ios::app : std::ios::binary | std::ios::trunc);
            if (!out) { werr = "cannot write " + file; ok = false; return false; }
            done = resumed ? have : 0;
            total = h.contentLength ? (resumed && h.contentLength < have ? h.contentLength + have : h.contentLength) : 0;
            return true;
        },
        [&](const char* p, size_t n) -> bool {
            out.write(p, (std::streamsize)n);
            if (!out) { werr = "write failed (disk full?)"; ok = false; return false; }
            done += n;
            int64_t now = util::nowMillis();
            if (now - lastTick > 200) {
                lastTick = now;
                PullProgress pp; pp.status = "downloading"; pp.total = total; pp.completed = done; cb(pp);
            }
            return true;
        }, r, err);
    out.close();
    if (!ok) { err = werr; return false; }
    if (!net) return false;   // err set by transport (partial file kept for resume)
    if (total && done < total) { err = "connection closed early; run the same command again to resume"; return false; }
    return true;
}

bool pullModel(Store& store, const ModelName& name, const std::string& sourceIn, const PullCallback& cb, std::string& err) {
    std::string source = sourceIn.empty() ? resolveSource(store, name) : sourceIn;
    if (source.empty()) {
        err = "no download source known for '" + name.full() + "'. Give one explicitly: `ollama pull " + name.full() +
              " http://host/model.gguf` (or a local .gguf path), add it to catalog.txt, or set OLLAMA_MIRROR";
        return false;
    }
    PullProgress pp; pp.status = "pulling manifest"; cb(pp);

    std::string file; bool temp = false;
    if (util::startsWith(source, "http://") || util::startsWith(source, "https://")) {
        if (!download(store, source, file, cb, err)) return false;
        temp = true;
    } else {
        file = source;
    }
    GgufInfo g = readGgufInfo(file);
    if (!g.valid) {
        err = "downloaded file is not a valid GGUF model";
        if (temp) util::removeFile(file);
        return false;
    }
    std::string hex;
    if (!store.importBlob(file, temp, hex, err, [&](const std::string& s, uint64_t d, uint64_t t) {
            PullProgress p; p.status = s; p.completed = d; p.total = t; cb(p);
        })) return false;

    ModelInfo m; m.name = name; m.blob = hex; m.size = (uint64_t)util::fileSize(store.blobPath(hex));
    m.modified = util::nowUnix();
    pp = PullProgress(); pp.status = "writing manifest"; cb(pp);
    if (!store.save(m, err)) return false;
    pp = PullProgress(); pp.status = "success"; cb(pp);
    return true;
}
