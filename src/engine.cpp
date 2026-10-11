#include "engine.h"

#include <algorithm>

void GenOptions::apply(const Json& j) {
    if (!j.isObject()) return;
    for (size_t i = 0; i < j.members().size(); i++) {
        const std::string& k = j.members()[i].first; const Json& v = j.members()[i].second;
        if (k == "num_ctx") num_ctx = (int)v.num(num_ctx);
        else if (k == "num_predict") num_predict = (int)v.num(num_predict);
        else if (k == "top_k") top_k = (int)v.num(top_k);
        else if (k == "top_p") top_p = (float)v.num(top_p);
        else if (k == "min_p") min_p = (float)v.num(min_p);
        else if (k == "temperature") temperature = (float)v.num(temperature);
        else if (k == "repeat_penalty") repeat_penalty = (float)v.num(repeat_penalty);
        else if (k == "repeat_last_n") repeat_last_n = (int)v.num(repeat_last_n);
        else if (k == "seed") seed = (int)v.num(seed);
        else if (k == "num_thread") num_thread = (int)v.num(num_thread);
        else if (k == "num_batch") num_batch = (int)v.num(num_batch);
        else if (k == "stop") {
            stop.clear();
            if (v.isArray()) for (size_t n = 0; n < v.items().size(); n++) stop.push_back(v.items()[n].str());
            else if (v.isString()) stop.push_back(v.str());
        }
    }
    if (num_ctx < 16) num_ctx = 16;
    if (num_batch < 8) num_batch = 8;
}

std::string StopFilter::feed(const std::string& piece) {
    if (stopped_) return "";
    pending_ += piece;
    // Full match: emit everything before the earliest stop sequence and stop.
    size_t best = std::string::npos;
    for (size_t i = 0; i < stops_.size(); i++) {
        if (stops_[i].empty()) continue;
        size_t p = pending_.find(stops_[i]);
        if (p != std::string::npos && p < best) best = p;
    }
    if (best != std::string::npos) {
        std::string out = pending_.substr(0, best);
        pending_.clear(); stopped_ = true; return out;
    }
    // Hold back the longest suffix that is a prefix of some stop sequence.
    size_t hold = 0;
    for (size_t i = 0; i < stops_.size(); i++) {
        const std::string& s = stops_[i];
        for (size_t l = std::min(s.size() - 1, pending_.size()); l > hold; l--)
            if (pending_.compare(pending_.size() - l, l, s, 0, l) == 0) { hold = l; break; }
    }
    std::string out = pending_.substr(0, pending_.size() - hold);
    pending_.erase(0, out.size());
    return out;
}

std::string StopFilter::flush() {
    if (stopped_) return "";
    std::string out = pending_; pending_.clear(); return out;
}

size_t utf8CompleteLen(const std::string& s) {
    size_t n = s.size();
    // Look back at most 3 bytes for an incomplete lead sequence.
    for (size_t back = 1; back <= 3 && back <= n; back++) {
        unsigned char c = (unsigned char)s[n - back];
        if ((c & 0xC0) == 0x80) continue;                    // continuation byte, keep looking
        size_t need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
        return need > back ? n - back : n;
    }
    return n;
}
