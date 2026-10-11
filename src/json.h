// Minimal JSON value / parser / serializer (C++11, no dependencies).
#pragma once
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

class Json {
public:
    enum Type { Null, Bool, Number, String, Array, Object };

    Json() : t_(Null), b_(false), n_(0) {}
    Json(bool v) : t_(Bool), b_(v), n_(0) {}
    Json(int v) : t_(Number), b_(false), n_(v) {}
    Json(long v) : t_(Number), b_(false), n_((double)v) {}
    Json(long long v) : t_(Number), b_(false), n_((double)v) {}
    Json(unsigned long long v) : t_(Number), b_(false), n_((double)v) {}
    Json(double v) : t_(Number), b_(false), n_(v) {}
    Json(const char* v) : t_(String), b_(false), n_(0), s_(v) {}
    Json(const std::string& v) : t_(String), b_(false), n_(0), s_(v) {}

    static Json array() { Json j; j.t_ = Array; return j; }
    static Json object() { Json j; j.t_ = Object; return j; }

    Type type() const { return t_; }
    bool isNull() const { return t_ == Null; }
    bool isObject() const { return t_ == Object; }
    bool isArray() const { return t_ == Array; }
    bool isString() const { return t_ == String; }
    bool isNumber() const { return t_ == Number; }
    bool isBool() const { return t_ == Bool; }

    // Object access. operator[] inserts a null member if missing.
    Json& operator[](const std::string& key);
    const Json* find(const std::string& key) const;
    bool has(const std::string& key) const { return find(key) != 0; }
    const std::vector<std::pair<std::string, Json> >& members() const { return o_; }

    // Array access.
    void push(const Json& v) { if (t_ != Array) { *this = array(); } a_.push_back(v); }
    const std::vector<Json>& items() const { return a_; }
    size_t size() const { return t_ == Array ? a_.size() : o_.size(); }

    // Typed getters with defaults.
    std::string str(const std::string& d = "") const { return t_ == String ? s_ : d; }
    double num(double d = 0) const { return t_ == Number ? n_ : d; }
    long long i64(long long d = 0) const { return t_ == Number ? (long long)n_ : d; }
    bool boolean(bool d = false) const { return t_ == Bool ? b_ : d; }

    std::string getStr(const std::string& k, const std::string& d = "") const {
        const Json* j = find(k); return j ? j->str(d) : d;
    }
    double getNum(const std::string& k, double d = 0) const {
        const Json* j = find(k); return j ? j->num(d) : d;
    }
    bool getBool(const std::string& k, bool d = false) const {
        const Json* j = find(k); return j ? j->boolean(d) : d;
    }

    std::string dump() const;
    // Returns false on syntax error (err gets a message).
    static bool parse(const std::string& text, Json& out, std::string* err = 0);

private:
    void dumpTo(std::string& out) const;
    Type t_;
    bool b_;
    double n_;
    std::string s_;
    std::vector<Json> a_;
    std::vector<std::pair<std::string, Json> > o_;
};

std::string jsonQuote(const std::string& s);
