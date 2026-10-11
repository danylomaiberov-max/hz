#include "json.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

Json& Json::operator[](const std::string& key) {
    if (t_ != Object) { *this = object(); }
    for (size_t i = 0; i < o_.size(); i++)
        if (o_[i].first == key) return o_[i].second;
    o_.push_back(std::make_pair(key, Json()));
    return o_.back().second;
}

const Json* Json::find(const std::string& key) const {
    if (t_ != Object) return 0;
    for (size_t i = 0; i < o_.size(); i++)
        if (o_[i].first == key) return &o_[i].second;
    return 0;
}

static void appendUtf8(std::string& out, unsigned cp) {
    if (cp < 0x80) out += (char)cp;
    else if (cp < 0x800) { out += (char)(0xC0 | (cp >> 6)); out += (char)(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) {
        out += (char)(0xE0 | (cp >> 12)); out += (char)(0x80 | ((cp >> 6) & 0x3F));
        out += (char)(0x80 | (cp & 0x3F));
    } else {
        out += (char)(0xF0 | (cp >> 18)); out += (char)(0x80 | ((cp >> 12) & 0x3F));
        out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F));
    }
}

std::string jsonQuote(const std::string& s) {
    std::string o = "\"";
    for (size_t i = 0; i < s.size(); i++) {
        unsigned char c = (unsigned char)s[i];
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            case '\b': o += "\\b"; break;
            case '\f': o += "\\f"; break;
            default:
                if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
                else o += (char)c;
        }
    }
    return o + "\"";
}

void Json::dumpTo(std::string& out) const {
    switch (t_) {
        case Null: out += "null"; break;
        case Bool: out += b_ ? "true" : "false"; break;
        case Number: {
            char b[48];
            if (n_ == std::floor(n_) && std::fabs(n_) < 9e15) std::snprintf(b, sizeof b, "%.0f", n_);
            else std::snprintf(b, sizeof b, "%.9g", n_);
            out += b; break;
        }
        case String: out += jsonQuote(s_); break;
        case Array:
            out += '[';
            for (size_t i = 0; i < a_.size(); i++) { if (i) out += ','; a_[i].dumpTo(out); }
            out += ']'; break;
        case Object:
            out += '{';
            for (size_t i = 0; i < o_.size(); i++) {
                if (i) out += ',';
                out += jsonQuote(o_[i].first); out += ':'; o_[i].second.dumpTo(out);
            }
            out += '}'; break;
    }
}

std::string Json::dump() const { std::string s; dumpTo(s); return s; }

namespace {
struct Parser {
    const std::string& s; size_t p; std::string err; int depth;
    explicit Parser(const std::string& str) : s(str), p(0), depth(0) {}

    void ws() { while (p < s.size() && (s[p] == ' ' || s[p] == '\t' || s[p] == '\n' || s[p] == '\r')) p++; }
    bool fail(const char* m) { if (err.empty()) { err = m; } return false; }

    bool hex4(unsigned& v) {
        if (p + 4 > s.size()) return false;
        v = 0;
        for (int i = 0; i < 4; i++) {
            char c = s[p++]; v <<= 4;
            if (c >= '0' && c <= '9') v |= c - '0';
            else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
            else return false;
        }
        return true;
    }

    bool str(std::string& o) {
        if (s[p] != '"') return fail("expected string");
        p++;
        while (p < s.size()) {
            char c = s[p++];
            if (c == '"') return true;
            if (c != '\\') { o += c; continue; }
            if (p >= s.size()) break;
            char e = s[p++];
            switch (e) {
                case '"': o += '"'; break; case '\\': o += '\\'; break; case '/': o += '/'; break;
                case 'b': o += '\b'; break; case 'f': o += '\f'; break; case 'n': o += '\n'; break;
                case 'r': o += '\r'; break; case 't': o += '\t'; break;
                case 'u': {
                    unsigned cp;
                    if (!hex4(cp)) return fail("bad \\u escape");
                    if (cp >= 0xD800 && cp < 0xDC00 && p + 1 < s.size() && s[p] == '\\' && s[p + 1] == 'u') {
                        p += 2; unsigned lo;
                        if (!hex4(lo)) return fail("bad \\u escape");
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    appendUtf8(o, cp); break;
                }
                default: return fail("bad escape");
            }
        }
        return fail("unterminated string");
    }

    bool value(Json& out) {
        ws();
        if (p >= s.size()) return fail("unexpected end");
        if (++depth > 64) return fail("too deep");
        bool ok = value2(out);
        depth--;
        return ok;
    }

    bool value2(Json& out) {
        char c = s[p];
        if (c == '{') {
            p++; out = Json::object(); ws();
            if (p < s.size() && s[p] == '}') { p++; return true; }
            for (;;) {
                ws(); std::string k;
                if (p >= s.size() || !str(k)) return fail("bad object key");
                ws();
                if (p >= s.size() || s[p++] != ':') return fail("expected ':'");
                Json v; if (!value(v)) return false;
                out[k] = v; ws();
                if (p >= s.size()) return fail("unexpected end");
                if (s[p] == ',') { p++; continue; }
                if (s[p] == '}') { p++; return true; }
                return fail("expected ',' or '}'");
            }
        }
        if (c == '[') {
            p++; out = Json::array(); ws();
            if (p < s.size() && s[p] == ']') { p++; return true; }
            for (;;) {
                Json v; if (!value(v)) return false;
                out.push(v); ws();
                if (p >= s.size()) return fail("unexpected end");
                if (s[p] == ',') { p++; continue; }
                if (s[p] == ']') { p++; return true; }
                return fail("expected ',' or ']'");
            }
        }
        if (c == '"') { std::string v; if (!str(v)) return false; out = Json(v); return true; }
        if (s.compare(p, 4, "true") == 0) { p += 4; out = Json(true); return true; }
        if (s.compare(p, 5, "false") == 0) { p += 5; out = Json(false); return true; }
        if (s.compare(p, 4, "null") == 0) { p += 4; out = Json(); return true; }
        if (c == '-' || (c >= '0' && c <= '9')) {
            const char* b = s.c_str() + p; char* e = 0;
            double d = std::strtod(b, &e);
            if (e == b) return fail("bad number");
            p += (size_t)(e - b); out = Json(d); return true;
        }
        return fail("unexpected character");
    }
};
}  // namespace

bool Json::parse(const std::string& text, Json& out, std::string* err) {
    Parser ps(text);
    Json v;
    if (!ps.value(v)) { if (err) *err = ps.err; return false; }
    ps.ws();
    if (ps.p != text.size()) { if (err) *err = "trailing data"; return false; }
    out = v;
    return true;
}
