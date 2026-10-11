// Unit tests for the portable core (no framework): json, sha256, names, Modelfile, templates, stop filter.
#include <cstdio>
#include <string>

#include "../src/engine.h"
#include "../src/json.h"
#include "../src/store.h"
#include "../src/template.h"
#include "../src/util.h"

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); fails++; } } while (0)
#define CHECK_EQ(a, b) do { std::string x = (a), y = (b); if (x != y) { std::printf("FAIL %s:%d\n  got:      [%s]\n  expected: [%s]\n", __FILE__, __LINE__, x.c_str(), y.c_str()); fails++; } } while (0)

static std::string render(const std::string& t, const std::string& sys, std::vector<Message> msgs) {
    TemplateInput in; in.system = sys; in.messages = msgs; std::string out, err;
    if (!renderTemplate(t, in, out, err)) return "ERR:" + err;
    return out;
}
static Message M(const char* r, const char* c) { Message m; m.role = r; m.content = c; return m; }

int main() {
    // JSON
    Json j; std::string e;
    CHECK(Json::parse("{\"a\":[1,2.5,\"x\\n\\u00e9\"],\"b\":{\"c\":true,\"d\":null}}", j, &e));
    CHECK_EQ(j.dump(), "{\"a\":[1,2.5,\"x\\n\xc3\xa9\"],\"b\":{\"c\":true,\"d\":null}}");
    CHECK(!Json::parse("{\"a\":", j, &e));
    CHECK(!Json::parse("[1,2] x", j, &e));

    // SHA-256
    CHECK_EQ(util::sha256Hex("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    CHECK_EQ(util::sha256Hex(""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");

    // Names
    ModelName n;
    CHECK(ModelName::parse("llama3.2", n, e)); CHECK_EQ(n.full(), "llama3.2:latest");
    CHECK(ModelName::parse("registry.ollama.ai/library/Phi3:mini", n, e)); CHECK_EQ(n.full(), "phi3:mini");
    CHECK(ModelName::parse("me/model:1b", n, e)); CHECK_EQ(n.full(), "me/model:1b");
    CHECK(!ModelName::parse("../etc/passwd", n, e));
    CHECK(!ModelName::parse("a b", n, e));

    // Modelfile
    Modelfile mf;
    CHECK(Modelfile::parse("# c\nFROM x.gguf\nPARAMETER stop \"a\"\nPARAMETER stop b\nPARAMETER num_ctx 512\n"
                           "SYSTEM \"\"\"line1\nline2\"\"\"\nTEMPLATE \"{{ .Prompt }}\"\nMESSAGE user hi\n", mf, e));
    CHECK_EQ(mf.from, "x.gguf"); CHECK_EQ(mf.system, "line1\nline2"); CHECK_EQ(mf.templ, "{{ .Prompt }}");
    CHECK(mf.messages.size() == 1);
    Json p = paramsToJson(mf.params);
    CHECK_EQ(p.dump(), "{\"stop\":[\"a\",\"b\"],\"num_ctx\":512}");
    CHECK(!Modelfile::parse("FROM x\nPARAMETER a", mf, e));
    CHECK(!Modelfile::parse("FROM x\nBOGUS y", mf, e));

    // Templates: legacy chatml
    CHECK_EQ(render(builtinTemplate("chatml"), "sys", std::vector<Message>(1, M("user", "hi"))),
             "<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n");
    // legacy with history
    std::vector<Message> h; h.push_back(M("user", "a")); h.push_back(M("assistant", "b")); h.push_back(M("user", "c"));
    CHECK_EQ(render(builtinTemplate("chatml"), "", h),
             "<|im_start|>user\na<|im_end|>\n<|im_start|>assistant\nb<|im_end|>\n<|im_start|>user\nc<|im_end|>\n<|im_start|>assistant\n");
    // messages-based template (llama3 style)
    const char* l3 = "{{- range .Messages }}<|start_header_id|>{{ .Role }}<|end_header_id|>\n\n{{ .Content }}<|eot_id|>{{ end }}"
                     "<|start_header_id|>assistant<|end_header_id|>\n\n";
    CHECK_EQ(render(l3, "S", h),
             "<|start_header_id|>system<|end_header_id|>\n\nS<|eot_id|><|start_header_id|>user<|end_header_id|>\n\na<|eot_id|>"
             "<|start_header_id|>assistant<|end_header_id|>\n\nb<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nc<|eot_id|>"
             "<|start_header_id|>assistant<|end_header_id|>\n\n");
    // if / else if / else, eq, $last
    const char* t2 = "{{ range $i, $_ := .Messages }}{{ $last := eq (len (slice $.Messages $i)) 1 }}"
                     "{{ if eq .Role \"user\" }}[U:{{ .Content }}]{{ else if eq .Role \"assistant\" }}[A:{{ .Content }}]"
                     "{{ else }}[S:{{ .Content }}]{{ end }}{{ if $last }}!{{ end }}{{ end }}";
    CHECK_EQ(render(t2, "s", h), "[S:s][U:a][A:b][U:c]!");
    // trim markers
    CHECK_EQ(render("a  {{- \" \" -}}  b{{ if not .System }}X{{ end }}", "", std::vector<Message>(1, M("user", "q"))), "a b" "X");
    CHECK(render("{{ .Nope }}", "", h).compare(0, 4, "ERR:") == 0);
    CHECK(render("{{ if .System }}x", "", h).compare(0, 4, "ERR:") == 0);

    // Stop filter
    std::vector<std::string> stops(1, "</s>");
    StopFilter f(stops);
    std::string o = f.feed("Hello <"); o += f.feed("/"); o += f.feed("s> ignored");
    CHECK_EQ(o, "Hello "); CHECK(f.stopped());
    StopFilter g(stops); std::string o2 = g.feed("a<"); o2 += g.feed("b"); o2 += g.flush();
    CHECK_EQ(o2, "a<b");
    // UTF-8 boundaries
    CHECK(utf8CompleteLen("ab\xe2\x82") == 2); CHECK(utf8CompleteLen("ab\xe2\x82\xac") == 5); CHECK(utf8CompleteLen("abc") == 3);

    // Options
    GenOptions go; Json oj; Json::parse("{\"temperature\":0.1,\"stop\":[\"x\"],\"num_ctx\":999}", oj);
    go.apply(oj); CHECK(go.num_ctx == 999 && go.stop.size() == 1);

    // Time parse
    CHECK(util::parseIso8601("2026-10-11T10:58:03Z") == 1791716283LL);

    std::printf(fails ? "%d FAILED\n" : "all tests passed\n", fails);
    return fails ? 1 : 0;
}
