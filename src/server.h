// Ollama-compatible REST API (/api/*) plus the OpenAI-compatible subset (/v1/*).
#pragma once
#include "http.h"
#include "runner.h"
#include "store.h"

#define OLLAMA_XP_VERSION "0.1.0-xp"

class ApiServer {
public:
    ApiServer(Store& store, Runner& runner) : store_(store), runner_(runner) {}
    void handle(const http::Request& rq, http::Response& rs);

private:
    void tags(http::Response& rs);
    void ps(http::Response& rs);
    void show(const Json& body, http::Response& rs);
    void generate(const Json& body, http::Response& rs);
    void chat(const Json& body, http::Response& rs);
    void embed(const Json& body, bool legacy, http::Response& rs);
    void pull(const Json& body, http::Response& rs);
    void create(const Json& body, http::Response& rs);
    void copy(const Json& body, http::Response& rs);
    void remove(const Json& body, http::Response& rs);
    void openaiChat(const Json& body, bool completionsApi, http::Response& rs);
    void openaiModels(http::Response& rs);
    void openaiEmbeddings(const Json& body, http::Response& rs);

    Store& store_;
    Runner& runner_;
};

Json modelDetails(const ModelInfo& m);
