// Local model store (ported layout idea from Ollama: content-addressed blobs + per-tag manifests)
// and Modelfile handling.
#pragma once
#include <stdint.h>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "json.h"

struct ModelName {
    std::string name;  // e.g. "llama3.2" or "user/model"
    std::string tag;   // e.g. "latest"
    std::string full() const { return name + ":" + tag; }
    static bool parse(const std::string& s, ModelName& out, std::string& err);
};

struct GgufInfo {
    bool valid = false;
    std::string architecture, name, sizeLabel, quantization;
    uint32_t fileType = 0;
    uint64_t contextLength = 0, embeddingLength = 0, blockCount = 0;
    std::string chatTemplate;
    uint64_t tensorCount = 0;
};
GgufInfo readGgufInfo(const std::string& path);

struct Message { std::string role, content; };

struct ModelInfo {
    ModelName name;
    std::string blob;       // sha256 hex of the weights file
    std::string blobPath;   // absolute path of the weights file
    uint64_t size = 0;
    int64_t modified = 0;
    std::string system, templ, license;
    Json params = Json::object();   // key -> number/string/array
    std::vector<Message> messages;
    GgufInfo gguf;
    std::string digest;     // short id (sha256 of manifest), like `ollama list` ID column
};

class Store {
public:
    explicit Store(const std::string& root);
    static std::string defaultRoot();   // %OLLAMA_MODELS% or ~/.ollama/models
    const std::string& root() const { return root_; }

    std::vector<ModelInfo> list();
    bool get(const ModelName& n, ModelInfo& out);
    bool save(ModelInfo& m, std::string& err);       // writes manifest
    bool remove(const ModelName& n, std::string& err);
    bool copy(const ModelName& from, const ModelName& to, std::string& err);

    std::string blobPath(const std::string& hex) const;
    // Move/copy a weights file into the blob store (hashing it). Returns hex digest.
    bool importBlob(const std::string& file, bool move, std::string& hex, std::string& err,
                    const std::function<void(const std::string&, uint64_t, uint64_t)>& progress = nullptr);
    std::string newTempPath() const;   // inside <root>/blobs, for downloads
    void pruneBlobs();                 // delete blobs no manifest references

private:
    std::string manifestPath(const ModelName& n) const;
    bool load(const std::string& manifestFile, ModelInfo& out);
    std::string root_;
};

struct Modelfile {
    std::string from, system, templ, license;
    std::vector<std::pair<std::string, std::string> > params;
    std::vector<Message> messages;
    std::string adapter;
    // Parses Modelfile text. Returns false with err on bad syntax.
    static bool parse(const std::string& text, Modelfile& out, std::string& err);
    std::string render() const;   // Modelfile text, used by `show --modelfile`
};

// Builds a model from a Modelfile (FROM <existing model> or FROM <file.gguf>).
bool createModel(Store& store, const ModelName& name, const Modelfile& mf, const std::string& baseDir,
                 std::string& err, const std::function<void(const std::string&)>& status);

Modelfile modelfileOf(const ModelInfo& m);
Json paramsToJson(const std::vector<std::pair<std::string, std::string> >& p);
