#include "util.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <string>
#include <iostream>

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#else
#include <sys/time.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace util {

std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) a++;
    while (b > a && std::isspace((unsigned char)s[b - 1])) b--;
    return s.substr(a, b - a);
}

std::string lower(const std::string& s) {
    std::string o = s;
    for (size_t i = 0; i < o.size(); i++) o[i] = (char)std::tolower((unsigned char)o[i]);
    return o;
}

bool startsWith(const std::string& s, const std::string& p) { return s.compare(0, p.size(), p) == 0; }
bool endsWith(const std::string& s, const std::string& p) {
    return s.size() >= p.size() && s.compare(s.size() - p.size(), p.size(), p) == 0;
}

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out; std::string cur;
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == sep) { out.push_back(cur); cur.clear(); } else cur += s[i];
    }
    out.push_back(cur);
    return out;
}

std::string replaceAll(std::string s, const std::string& from, const std::string& to) {
    if (from.empty()) return s;
    size_t p = 0;
    while ((p = s.find(from, p)) != std::string::npos) { s.replace(p, from.size(), to); p += to.size(); }
    return s;
}

std::string humanBytes(uint64_t n) {
    char b[48];
    if (n >= 1000000000ULL) std::snprintf(b, sizeof b, "%.1f GB", n / 1e9);
    else if (n >= 1000000ULL) std::snprintf(b, sizeof b, n >= 100000000ULL ? "%.0f MB" : "%.1f MB", n / 1e6);
    else if (n >= 1000ULL) std::snprintf(b, sizeof b, "%.0f KB", n / 1e3);
    else std::snprintf(b, sizeof b, "%llu B", (unsigned long long)n);
    return b;
}

static std::string plural(long long v, const char* unit) {
    char b[64];
    std::snprintf(b, sizeof b, "%lld %s%s", v, unit, v == 1 ? "" : "s");
    return b;
}

static std::string span(int64_t sec) {
    if (sec < 5) return "Less than a second";
    if (sec < 60) return plural(sec, "second");
    if (sec < 3600) return sec < 120 ? "About a minute" : plural(sec / 60, "minute");
    if (sec < 86400) return sec < 7200 ? "About an hour" : plural(sec / 3600, "hour");
    if (sec < 7 * 86400) return plural(sec / 86400, "day");
    if (sec < 30 * 86400) return plural(sec / (7 * 86400), "week");
    if (sec < 365 * 86400) return plural(sec / (30 * 86400), "month");
    return plural(sec / (365 * 86400), "year");
}

std::string humanAgo(int64_t t) {
    int64_t d = nowUnix() - t;
    if (d < 5) return "Less than a second ago";
    return span(d) + " ago";
}

std::string humanFromNow(int64_t t) {
    int64_t d = t - nowUnix();
    if (d <= 0) return "Expiring";
    return span(d) + " from now";
}

std::string iso8601(int64_t t) {
    time_t tt = (time_t)t; struct tm g;
#ifdef _WIN32
    struct tm* r = gmtime(&tt); if (r) g = *r; else std::memset(&g, 0, sizeof g);
#else
    gmtime_r(&tt, &g);
#endif
    char b[40];
    std::strftime(b, sizeof b, "%Y-%m-%dT%H:%M:%SZ", &g);
    return b;
}

int64_t parseIso8601(const std::string& s) {
    int Y, M, D, h, m, sec;
    if (std::sscanf(s.c_str(), "%d-%d-%dT%d:%d:%d", &Y, &M, &D, &h, &m, &sec) != 6) return 0;
    // days from civil (H. Hinnant)
    Y -= M <= 2;
    int64_t era = (Y >= 0 ? Y : Y - 399) / 400;
    int64_t yoe = Y - era * 400;
    int64_t doy = (153 * (M + (M > 2 ? -3 : 9)) + 2) / 5 + D - 1;
    int64_t doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    int64_t days = era * 146097 + doe - 719468;
    return days * 86400 + h * 3600 + m * 60 + sec;
}

int64_t nowUnix() { return (int64_t)std::time(0); }

int64_t nowMillis() {
#ifdef _WIN32
    // GetTickCount64 is Vista+; use FILETIME which exists on XP.
    FILETIME ft; GetSystemTimeAsFileTime(&ft);
    uint64_t v = ((uint64_t)ft.dwHighDateTime << 32) | ft.dwLowDateTime;
    return (int64_t)((v - 116444736000000000ULL) / 10000ULL);
#else
    struct timeval tv; gettimeofday(&tv, 0);
    return (int64_t)tv.tv_sec * 1000 + tv.tv_usec / 1000;
#endif
}

std::string getEnv(const char* name, const std::string& def) {
    const char* v = std::getenv(name);
    return (v && *v) ? std::string(v) : def;
}

std::string homeDir() {
#ifdef _WIN32
    std::string h = getEnv("USERPROFILE");
    if (!h.empty()) return h;
    std::string d = getEnv("HOMEDRIVE"), p = getEnv("HOMEPATH");
    if (!d.empty() || !p.empty()) return d + p;
    return "C:";
#else
    return getEnv("HOME", "/tmp");
#endif
}

static fs::path P(const std::string& s) { return fs::u8path(s); }

bool fileExists(const std::string& path) {
    std::error_code ec; return fs::is_regular_file(P(path), ec);
}
bool dirExists(const std::string& path) {
    std::error_code ec; return fs::is_directory(P(path), ec);
}
bool makeDirs(const std::string& path) {
    std::error_code ec; fs::create_directories(P(path), ec); return dirExists(path);
}
bool removeFile(const std::string& path) {
    std::error_code ec; return fs::remove(P(path), ec);
}
bool removeEmptyDirs(const std::string& root, const std::string& stop) {
    std::error_code ec;
    fs::path cur = P(root), end = P(stop);
    while (!cur.empty() && cur != end && fs::is_directory(cur, ec) && fs::is_empty(cur, ec)) {
        fs::remove(cur, ec); cur = cur.parent_path();
    }
    return true;
}
bool copyFile(const std::string& from, const std::string& to) {
    std::error_code ec;
    fs::create_directories(P(to).parent_path(), ec);
    return fs::copy_file(P(from), P(to), fs::copy_options::overwrite_existing, ec);
}
bool moveFile(const std::string& from, const std::string& to) {
    std::error_code ec;
    fs::create_directories(P(to).parent_path(), ec);
    fs::rename(P(from), P(to), ec);
    if (!ec) return true;
    // Cross-volume: copy then delete.
    if (copyFile(from, to)) { removeFile(from); return true; }
    return false;
}
int64_t fileSize(const std::string& path) {
    std::error_code ec; auto n = fs::file_size(P(path), ec); return ec ? -1 : (int64_t)n;
}
bool readFile(const std::string& path, std::string& out) {
    std::ifstream f(P(path), std::ios::binary);
    if (!f) return false;
    out.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return true;
}
bool writeFile(const std::string& path, const std::string& data) {
    std::error_code ec; fs::create_directories(P(path).parent_path(), ec);
    std::ofstream f(P(path), std::ios::binary | std::ios::trunc);
    if (!f) return false;
    f.write(data.data(), (std::streamsize)data.size());
    return (bool)f;
}
std::vector<std::string> listFiles(const std::string& root) {
    std::vector<std::string> out; std::error_code ec;
    if (!fs::is_directory(P(root), ec)) return out;
    for (fs::recursive_directory_iterator it(P(root), ec), end; !ec && it != end; it.increment(ec)) {
        std::error_code e2;
        if (it->is_regular_file(e2)) out.push_back(it->path().generic_u8string());
    }
    std::sort(out.begin(), out.end());
    return out;
}
std::string joinPath(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    char c = a[a.size() - 1];
    return (c == '/' || c == '\\') ? a + b : a + "/" + b;
}

// ---- SHA-256 ----
namespace {
struct Sha256 {
    uint32_t h[8]; uint64_t len; unsigned char buf[64]; size_t n;
    Sha256() { reset(); }
    void reset() {
        static const uint32_t init[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                         0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
        std::memcpy(h, init, sizeof h); len = 0; n = 0;
    }
    static uint32_t rotr(uint32_t x, int c) { return (x >> c) | (x << (32 - c)); }
    void block(const unsigned char* p) {
        static const uint32_t k[64] = {
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
            0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
            0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
            0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
            0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
            0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        uint32_t w[64];
        for (int i = 0; i < 16; i++)
            w[i] = ((uint32_t)p[i*4] << 24) | ((uint32_t)p[i*4+1] << 16) | ((uint32_t)p[i*4+2] << 8) | p[i*4+3];
        for (int i = 16; i < 64; i++) {
            uint32_t s0 = rotr(w[i-15],7) ^ rotr(w[i-15],18) ^ (w[i-15] >> 3);
            uint32_t s1 = rotr(w[i-2],17) ^ rotr(w[i-2],19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for (int i = 0; i < 64; i++) {
            uint32_t S1 = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25);
            uint32_t ch = (e & f) ^ (~e & g);
            uint32_t t1 = hh + S1 + ch + k[i] + w[i];
            uint32_t S0 = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22);
            uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t t2 = S0 + mj;
            hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=hh;
    }
    void update(const unsigned char* p, size_t m) {
        len += m;
        while (m) {
            size_t t = std::min(m, 64 - n);
            std::memcpy(buf + n, p, t); n += t; p += t; m -= t;
            if (n == 64) { block(buf); n = 0; }
        }
    }
    std::string finish() {
        uint64_t bits = len * 8;
        unsigned char pad = 0x80; update(&pad, 1);
        unsigned char z = 0; while (n != 56) update(&z, 1);
        unsigned char l[8]; for (int i = 0; i < 8; i++) l[i] = (unsigned char)(bits >> (56 - 8 * i));
        update(l, 8);
        char out[65];
        for (int i = 0; i < 8; i++) std::snprintf(out + i * 8, 9, "%08x", h[i]);
        return out;
    }
};
}  // namespace

std::string sha256Hex(const std::string& data) {
    Sha256 s; s.update((const unsigned char*)data.data(), data.size()); return s.finish();
}

std::string sha256File(const std::string& path, const std::function<void(uint64_t, uint64_t)>& progress) {
    std::ifstream f(P(path), std::ios::binary);
    if (!f) return "";
    int64_t total = fileSize(path);
    Sha256 s; std::vector<char> buf(1 << 20); uint64_t done = 0;
    while (f) {
        f.read(buf.data(), (std::streamsize)buf.size());
        std::streamsize got = f.gcount();
        if (got <= 0) break;
        s.update((const unsigned char*)buf.data(), (size_t)got);
        done += (uint64_t)got;
        if (progress) progress(done, total < 0 ? 0 : (uint64_t)total);
    }
    return s.finish();
}

// ---- Console ----
bool stdinIsTty() {
#ifdef _WIN32
    return _isatty(0) != 0;
#else
    return isatty(0) != 0;
#endif
}
bool stdoutIsTty() {
#ifdef _WIN32
    return _isatty(1) != 0;
#else
    return isatty(1) != 0;
#endif
}

void setupConsoleUtf8() {
#ifdef _WIN32
    // Console I/O goes through ReadConsoleW/WriteConsoleW (consoleWrite/readLine); keep the code page for redirected output.
    SetConsoleOutputCP(65001);
#endif
}

void consoleWrite(const std::string& text) {
#ifdef _WIN32
    static HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode;
    if (GetConsoleMode(h, &mode)) {
        std::fflush(stdout);
        int n = MultiByteToWideChar(CP_UTF8, 0, text.data(), (int)text.size(), 0, 0);
        if (n > 0) {
            std::wstring w((size_t)n, L'\0');
            MultiByteToWideChar(CP_UTF8, 0, text.data(), (int)text.size(), &w[0], n);
            DWORD written;
            WriteConsoleW(h, w.data(), (DWORD)w.size(), &written, 0);
        }
        return;
    }
#endif
    std::fwrite(text.data(), 1, text.size(), stdout);
    std::fflush(stdout);
}

bool readLine(const std::string& prompt, std::string& line) {
    if (!prompt.empty()) consoleWrite(prompt);
    line.clear();
#ifdef _WIN32
    HANDLE h = GetStdHandle(STD_INPUT_HANDLE);
    DWORD mode;
    if (GetConsoleMode(h, &mode)) {
        std::wstring w; wchar_t buf[512]; DWORD got;
        for (;;) {
            if (!ReadConsoleW(h, buf, 512, &got, 0) || got == 0) { if (w.empty()) return false; break; }
            w.append(buf, got);
            if (w[w.size() - 1] == L'\n') break;
        }
        while (!w.empty() && (w[w.size() - 1] == L'\n' || w[w.size() - 1] == L'\r')) w.erase(w.size() - 1);
        if (!w.empty()) {
            int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), 0, 0, 0, 0);
            line.assign((size_t)n, '\0');
            WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), &line[0], n, 0, 0);
        }
        return true;
    }
#endif
    if (!std::getline(std::cin, line)) return false;
    if (!line.empty() && line[line.size() - 1] == '\r') line.erase(line.size() - 1);
    return true;
}

}  // namespace util
