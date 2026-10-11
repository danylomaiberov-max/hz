// `pull`: download a GGUF over plain HTTP (resumable) or import a local file as a model.
#pragma once
#include <functional>
#include <string>

#include "store.h"

struct PullProgress {
    std::string status;     // "pulling manifest", "downloading", "verifying sha256 digest", "success" ...
    std::string digest;
    uint64_t total = 0, completed = 0;
};
typedef std::function<void(const PullProgress&)> PullCallback;

// Where `source` can be: http:// URL to a .gguf, a local .gguf path, or empty (then the catalog /
// OLLAMA_MIRROR is consulted for `name`). Returns false with err on failure.
bool pullModel(Store& store, const ModelName& name, const std::string& source, const PullCallback& cb, std::string& err);

// Looks up `name` in <root>/../catalog.txt ("name:tag  url" per line) or OLLAMA_MIRROR
// ("http://host/path/{name}-{tag}.gguf"). Empty if nothing known.
std::string resolveSource(const Store& store, const ModelName& name);
