#include "template.h"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>

#include "util.h"

namespace {

struct Node {
    enum Kind { Text, Print, If, Range, Assign } kind = Text;
    std::string text;                  // Text / expression / assignment var name
    std::vector<std::string> cond;     // tokens of condition (If/Print/Assign rhs)
    std::vector<Node> body, other;     // if-body / else-body ; range body
};

typedef std::vector<std::string> Toks;

// Splits an action into tokens: words, quoted strings (kept with quotes), parentheses.
Toks tokenize(const std::string& s) {
    Toks t; size_t i = 0;
    while (i < s.size()) {
        char c = s[i];
        if (std::isspace((unsigned char)c)) { i++; continue; }
        if (c == '(' || c == ')' || c == ',') { t.push_back(std::string(1, c)); i++; continue; }
        if (c == ':' && i + 1 < s.size() && s[i + 1] == '=') { t.push_back(":="); i += 2; continue; }
        if (c == '"') {
            std::string v = "\""; i++;
            while (i < s.size() && s[i] != '"') {
                if (s[i] == '\\' && i + 1 < s.size()) {
                    i++; char e = s[i];
                    v += e == 'n' ? '\n' : e == 't' ? '\t' : e;
                } else v += s[i];
                i++;
            }
            i++; t.push_back(v + "\""); continue;
        }
        size_t b = i;
        while (i < s.size() && !std::isspace((unsigned char)s[i]) && s[i] != '(' && s[i] != ')' && s[i] != ',' &&
               !(s[i] == ':' && i + 1 < s.size() && s[i + 1] == '=')) i++;
        t.push_back(s.substr(b, i - b));
    }
    return t;
}

struct ParseState {
    struct Piece { bool action; std::string text; };
    std::vector<Piece> pieces; size_t pos = 0; std::string err;
};

bool lex(const std::string& tmpl, ParseState& ps) {
    size_t i = 0; std::string text;
    while (i < tmpl.size()) {
        size_t a = tmpl.find("{{", i);
        if (a == std::string::npos) { text += tmpl.substr(i); break; }
        text += tmpl.substr(i, a - i);
        size_t b = tmpl.find("}}", a);
        if (b == std::string::npos) { ps.err = "unclosed action"; return false; }
        std::string act = tmpl.substr(a + 2, b - a - 2);
        bool trimL = !act.empty() && act[0] == '-' && act.size() > 1 && std::isspace((unsigned char)act[1]);
        bool trimR = act.size() > 1 && act[act.size() - 1] == '-' && std::isspace((unsigned char)act[act.size() - 2]);
        if (trimL) { act = act.substr(1); while (!text.empty() && std::isspace((unsigned char)text[text.size() - 1])) text.erase(text.size() - 1); }
        if (trimR) act = act.substr(0, act.size() - 1);
        if (!text.empty()) { ParseState::Piece p = {false, text}; ps.pieces.push_back(p); text.clear(); }
        ParseState::Piece p = {true, util::trim(act)}; ps.pieces.push_back(p);
        i = b + 2;
        if (trimR) while (i < tmpl.size() && std::isspace((unsigned char)tmpl[i])) i++;
    }
    if (!text.empty()) { ParseState::Piece p = {false, text}; ps.pieces.push_back(p); }
    return true;
}

bool parseBlock(ParseState& ps, std::vector<Node>& out, std::string& stop, Toks& stopToks);

// Parses the body of an if/range (and its else / else-if chain) up to the shared {{ end }}.
bool parseIf(ParseState& ps, Node& n) {
    std::string st; Toks stt;
    if (!parseBlock(ps, n.body, st, stt)) return false;
    if (st == "else") {
        if (stt.size() > 1 && stt[1] == "if") {
            Node e; e.kind = Node::If; e.cond.assign(stt.begin() + 2, stt.end());
            if (!parseIf(ps, e)) return false;
            n.other.push_back(e); return true;
        }
        std::string st2; Toks s2;
        if (!parseBlock(ps, n.other, st2, s2)) return false;
        st = st2;
    }
    if (st != "end") { ps.err = "missing {{ end }}"; return false; }
    return true;
}

// Parses nodes until an {{ end }} / {{ else }} (returned via `stop`).
bool parseBlock(ParseState& ps, std::vector<Node>& out, std::string& stop, Toks& stopToks) {
    stop.clear();
    while (ps.pos < ps.pieces.size()) {
        const ParseState::Piece& pc = ps.pieces[ps.pos++];
        if (!pc.action) { Node n; n.kind = Node::Text; n.text = pc.text; out.push_back(n); continue; }
        Toks t = tokenize(pc.text);
        if (t.empty()) continue;
        if (t[0] == "end") { stop = "end"; return true; }
        if (t[0] == "else") { stop = "else"; stopToks = t; return true; }
        if (t[0][0] == '/' ) continue;   // comment-ish
        if (t[0] == "if" || t[0] == "range") {
            Node n; n.kind = t[0] == "if" ? Node::If : Node::Range;
            n.cond.assign(t.begin() + 1, t.end());
            if (!parseIf(ps, n)) return false;
            out.push_back(n); continue;
        }
        if (t.size() >= 3 && t[0][0] == '$' && t[1] == ":=") {
            Node n; n.kind = Node::Assign; n.text = t[0]; n.cond.assign(t.begin() + 2, t.end());
            out.push_back(n); continue;
        }
        if (t[0] == "define" || t[0] == "template" || t[0] == "with" || t[0] == "block") {
            ps.err = "unsupported action: " + t[0]; return false;
        }
        Node n; n.kind = Node::Print; n.cond = t; out.push_back(n);
    }
    stop.clear();
    return true;
}

struct Ctx {
    std::map<std::string, std::string> vars;   // System, Prompt, Response, Role, Content
    const std::vector<Message>* msgs = 0;
    int index = -1;                             // inside range
    bool hasMsgs = false;
    std::map<std::string, bool> flags;          // $last etc.
    std::string err;
};

struct Val { std::string s; bool isStr = true; bool b = false; };

bool truthy(const Val& v) { return v.isStr ? !v.s.empty() : v.b; }

Val evalExpr(const Toks& t, size_t& i, Ctx& c);

Val evalTerm(const Toks& t, size_t& i, Ctx& c) {
    Val v;
    if (i >= t.size()) return v;
    const std::string& w = t[i];
    if (w == "(") {
        i++; v = evalExpr(t, i, c);
        if (i < t.size() && t[i] == ")") i++;
        return v;
    }
    i++;
    if (w[0] == '"') { v.s = w.substr(1, w.size() - 2); return v; }
    if (w == "true" || w == "false") { v.isStr = false; v.b = w == "true"; return v; }
    if (w[0] == '.' || (w[0] == '$' && w.size() > 1 && w[1] == '.')) {
        std::string name = w[0] == '$' ? w.substr(2) : w.substr(1);
        bool root = w[0] == '$';
        if (name == "Messages") { v.isStr = false; v.b = c.msgs && !c.msgs->empty(); return v; }
        if (!root && c.index >= 0 && c.msgs) {
            const Message& m = (*c.msgs)[(size_t)c.index];
            if (name == "Role") { v.s = m.role; return v; }
            if (name == "Content") { v.s = m.content; return v; }
        }
        std::map<std::string, std::string>::iterator f = c.vars.find(name);
        if (f != c.vars.end()) v.s = f->second;
        else if (name != "Tools" && name != "ToolCalls" && name != "Think" && name != "IsThinkSet" && name != "Suffix" &&
                 name != "Role" && name != "Content" && name != "ThinkLevel")
            c.err = "unknown field ." + name;
        return v;
    }
    if (w[0] == '$') {
        std::map<std::string, bool>::iterator f = c.flags.find(w);
        if (f != c.flags.end()) { v.isStr = false; v.b = f->second; return v; }
        v.s = ""; return v;
    }
    if (w == "not") { Val a = evalTerm(t, i, c); v.isStr = false; v.b = !truthy(a); return v; }
    if (w == "len" || w == "slice") { Val a = evalTerm(t, i, c); (void)a; return v; }
    // multi-arg functions
    if (w == "eq" || w == "ne" || w == "and" || w == "or") {
        Val a = evalTerm(t, i, c), b = evalTerm(t, i, c);
        v.isStr = false;
        if (w == "eq") v.b = a.s == b.s && (a.isStr == b.isStr ? (a.isStr || a.b == b.b) : false);
        else if (w == "ne") v.b = !(a.s == b.s && (a.isStr == b.isStr ? (a.isStr || a.b == b.b) : false));
        else if (w == "and") v.b = truthy(a) && truthy(b);
        else v.b = truthy(a) || truthy(b);
        return v;
    }
    // number literal
    v.s = w;
    return v;
}

Val evalExpr(const Toks& t, size_t& i, Ctx& c) { return evalTerm(t, i, c); }

void run(const std::vector<Node>& nodes, Ctx& c, std::string& out);

bool isLastExpr(const Toks& t) {
    for (size_t i = 0; i < t.size(); i++) if (t[i] == "slice" || t[i] == "len") return true;
    return false;
}

void run(const std::vector<Node>& nodes, Ctx& c, std::string& out) {
    for (size_t k = 0; k < nodes.size() && c.err.empty(); k++) {
        const Node& n = nodes[k];
        switch (n.kind) {
            case Node::Text: out += n.text; break;
            case Node::Print: {
                size_t i = 0; Val v = evalExpr(n.cond, i, c);
                if (v.isStr) out += v.s;
                break;
            }
            case Node::Assign: {
                if (isLastExpr(n.cond)) c.flags[n.text] = c.msgs && c.index >= 0 && (size_t)c.index + 1 == c.msgs->size();
                else { size_t i = 0; Val v = evalExpr(n.cond, i, c); c.flags[n.text] = truthy(v); }
                break;
            }
            case Node::If: {
                size_t i = 0; Val v = evalExpr(n.cond, i, c);
                run(truthy(v) ? n.body : n.other, c, out);
                break;
            }
            case Node::Range: {
                // Only `range .Messages` (optionally with `$i, $m :=`) is supported.
                bool isMsgs = false;
                for (size_t i = 0; i < n.cond.size(); i++) if (n.cond[i] == ".Messages" || n.cond[i] == "$.Messages") isMsgs = true;
                if (!isMsgs || !c.msgs) { c.err = "unsupported range"; break; }
                int saved = c.index;
                for (size_t m = 0; m < c.msgs->size(); m++) { c.index = (int)m; run(n.body, c, out); }
                c.index = saved;
                break;
            }
        }
    }
}

bool usesMessages(const std::vector<Node>& nodes) {
    for (size_t i = 0; i < nodes.size(); i++) {
        const Node& n = nodes[i];
        for (size_t k = 0; k < n.cond.size(); k++) if (n.cond[k].find("Messages") != std::string::npos) return true;
        if (usesMessages(n.body) || usesMessages(n.other)) return true;
    }
    return false;
}

const char* kResp = "\x01RESPONSE\x01";

}  // namespace

bool renderTemplate(const std::string& tmpl, const TemplateInput& in, std::string& out, std::string& err) {
    ParseState ps;
    if (!lex(tmpl, ps)) { err = ps.err; return false; }
    std::vector<Node> nodes; std::string stop; Toks st;
    if (!parseBlock(ps, nodes, stop, st) || !stop.empty()) { err = ps.err.empty() ? "unexpected {{ end }}" : ps.err; return false; }

    std::vector<Message> msgs = in.messages;
    if (!in.system.empty() && (msgs.empty() || msgs[0].role != "system")) {
        Message s; s.role = "system"; s.content = in.system; msgs.insert(msgs.begin(), s);
    }

    if (usesMessages(nodes)) {
        Ctx c; c.msgs = &msgs; c.hasMsgs = true;
        for (size_t i = 0; i < msgs.size(); i++) if (msgs[i].role == "system") { c.vars["System"] = msgs[i].content; break; }
        std::string o; run(nodes, c, o);
        if (!c.err.empty()) { err = c.err; return false; }
        out = o; return true;
    }

    // Legacy (.System/.Prompt/.Response) template: render one pass per user turn.
    std::string sys, result;
    size_t i = 0;
    while (i < msgs.size()) {
        Message cur = msgs[i];
        if (cur.role == "system") { sys += (sys.empty() ? "" : "\n") + cur.content; i++; if (i < msgs.size() && msgs[i].role != "user") continue; if (i >= msgs.size()) break; cur = msgs[i]; }
        std::string reply; bool hasReply = false;
        if (i + 1 < msgs.size() && msgs[i + 1].role == "assistant") { reply = msgs[i + 1].content; hasReply = true; }
        Ctx c;
        c.vars["System"] = sys; c.vars["Prompt"] = cur.content; c.vars["Response"] = kResp;
        std::string o; run(nodes, c, o);
        if (!c.err.empty()) { err = c.err; return false; }
        size_t at = o.find(kResp);
        if (hasReply) { if (at != std::string::npos) o.replace(at, std::strlen(kResp), reply); i += 2; }
        else { if (at != std::string::npos) o = in.addGenerationPrompt ? o.substr(0, at) : util::replaceAll(o, kResp, ""); i += 1; }
        result += o; sys.clear();
    }
    out = result;
    return true;
}

std::string chatmlPrompt(const std::vector<Message>& msgs, bool addAssistant) {
    std::string o;
    for (size_t i = 0; i < msgs.size(); i++) o += "<|im_start|>" + msgs[i].role + "\n" + msgs[i].content + "<|im_end|>\n";
    if (addAssistant) o += "<|im_start|>assistant\n";
    return o;
}

std::string builtinTemplate(const std::string& name) {
    if (name == "chatml")
        return "{{ if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}{{ if .Prompt }}<|im_start|>user\n{{ .Prompt }}<|im_end|>\n{{ end }}<|im_start|>assistant\n{{ .Response }}<|im_end|>\n";
    if (name == "llama3")
        return "{{ if .System }}<|start_header_id|>system<|end_header_id|>\n\n{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>\n\n{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>\n\n{{ .Response }}<|eot_id|>";
    if (name == "alpaca")
        return "{{ if .System }}{{ .System }}\n\n{{ end }}{{ if .Prompt }}### Instruction:\n{{ .Prompt }}\n\n{{ end }}### Response:\n{{ .Response }}";
    if (name == "zephyr")
        return "{{ if .System }}<|system|>\n{{ .System }}</s>\n{{ end }}{{ if .Prompt }}<|user|>\n{{ .Prompt }}</s>\n{{ end }}<|assistant|>\n{{ .Response }}</s>\n";
    return "";
}
