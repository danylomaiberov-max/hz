// Subset of Go text/template as used by Ollama model TEMPLATEs.
// Supported: {{ .System }} {{ .Prompt }} {{ .Response }} {{ .Messages }} loops (range), if/else if/else/end,
// not/and/or/eq/ne, string literals, whitespace trim markers ({{- -}}), $.Var, $last-style assignments.
#pragma once
#include <string>
#include <vector>

#include "store.h"

struct TemplateInput {
    std::string system;
    std::vector<Message> messages;   // full conversation, last one is usually the user's turn
    bool addGenerationPrompt = true;
};

// Renders the prompt text for the model. Returns false (with err) if the template is not understood.
bool renderTemplate(const std::string& tmpl, const TemplateInput& in, std::string& out, std::string& err);

// Used when a model has no TEMPLATE: plain ChatML.
std::string chatmlPrompt(const std::vector<Message>& msgs, bool addAssistant);
// Templates seen in the wild, by name; empty if unknown.
std::string builtinTemplate(const std::string& name);
