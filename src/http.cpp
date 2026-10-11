#include "http.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#include "util.h"

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
typedef SOCKET sock_t;
#define SOCK_INVALID INVALID_SOCKET
static void closeSock(sock_t s) { closesocket(s); }
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>
typedef int sock_t;
#define SOCK_INVALID (-1)
static void closeSock(sock_t s) { ::close(s); }
#endif

namespace http {

bool startup() {
#ifdef _WIN32
    static bool done = false, ok = false;
    if (!done) { WSADATA d; ok = WSAStartup(MAKEWORD(2, 2), &d) == 0; done = true; }
    return ok;
#else
    signal(SIGPIPE, SIG_IGN);
    return true;
#endif
}

static const char* reason(int s) {
    switch (s) {
        case 200: return "OK"; case 204: return "No Content"; case 400: return "Bad Request";
        case 404: return "Not Found"; case 405: return "Method Not Allowed"; case 500: return "Internal Server Error";
        case 501: return "Not Implemented"; case 413: return "Payload Too Large"; default: return "OK";
    }
}

// ---------- Response ----------
bool Response::sendRaw(const char* p, size_t n) {
    if (failed_) return false;
    while (n) {
        int w = (int)::send((sock_t)sock_, p, (int)std::min<size_t>(n, 1 << 16), 0);
        if (w <= 0) { failed_ = true; return false; }
        p += w; n -= (size_t)w;
    }
    return true;
}

static std::string headCommon(int status, const std::string& ct) {
    char b[160];
    std::snprintf(b, sizeof b, "HTTP/1.1 %d %s\r\n", status, reason(status));
    return std::string(b) + "Content-Type: " + ct + "\r\nAccess-Control-Allow-Origin: *\r\n"
           "Access-Control-Allow-Methods: GET, POST, DELETE, HEAD, OPTIONS\r\nAccess-Control-Allow-Headers: *\r\n"
           "Connection: close\r\n";
}

bool Response::startStream(int status, const std::string& ct) {
    std::string h = headCommon(status, ct) + "Transfer-Encoding: chunked\r\n\r\n";
    headSent_ = true; chunked_ = true;
    return sendRaw(h.data(), h.size());
}

bool Response::write(const std::string& data) {
    if (data.empty()) return !failed_;
    char b[24]; std::snprintf(b, sizeof b, "%zx\r\n", data.size());
    std::string frame = std::string(b) + data + "\r\n";
    return sendRaw(frame.data(), frame.size());
}

void Response::finish() { if (chunked_) sendRaw("0\r\n\r\n", 5); }

void Response::send(int status, const std::string& ct, const std::string& body) {
    char b[48]; std::snprintf(b, sizeof b, "Content-Length: %zu\r\n\r\n", body.size());
    std::string h = headCommon(status, ct) + b;
    headSent_ = true;
    sendRaw(h.data(), h.size());
    sendRaw(body.data(), body.size());
}

// ---------- Server ----------
Server::Server() : sock_((intptr_t)SOCK_INVALID), stopping_(false) {}
Server::~Server() { stop(); }

bool Server::listen(const std::string& host, int port, std::string& err) {
    startup();
    sock_t s = ::socket(AF_INET, SOCK_STREAM, 0);
    if (s == SOCK_INVALID) { err = "socket() failed"; return false; }
    int one = 1;
#ifndef _WIN32
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, (const char*)&one, sizeof one);
#endif
    sockaddr_in a; std::memset(&a, 0, sizeof a);
    a.sin_family = AF_INET; a.sin_port = htons((unsigned short)port);
    a.sin_addr.s_addr = (host.empty() || host == "0.0.0.0") ? htonl(INADDR_ANY) : inet_addr(host == "localhost" ? "127.0.0.1" : host.c_str());
    if (a.sin_addr.s_addr == INADDR_NONE) { err = "bad host address: " + host; closeSock(s); return false; }
    if (::bind(s, (sockaddr*)&a, sizeof a) != 0) { err = "cannot bind " + host + ":" + std::to_string(port) + " (already in use?)"; closeSock(s); return false; }
    if (::listen(s, 16) != 0) { err = "listen() failed"; closeSock(s); return false; }
    sock_ = (intptr_t)s;
    return true;
}

void Server::stop() {
    stopping_ = true;
    if (sock_ != (intptr_t)SOCK_INVALID) { closeSock((sock_t)sock_); sock_ = (intptr_t)SOCK_INVALID; }
}

static bool readHead(sock_t s, std::string& buf, size_t& headEnd) {
    char tmp[4096];
    while ((headEnd = buf.find("\r\n\r\n")) == std::string::npos) {
        if (buf.size() > (1 << 16)) return false;
        int n = (int)::recv(s, tmp, sizeof tmp, 0);
        if (n <= 0) return false;
        buf.append(tmp, (size_t)n);
    }
    return true;
}

static void handleConn(sock_t s, Handler h, std::string peer) {
    Response resp((intptr_t)s);
    std::string buf; size_t he = 0;
    if (readHead(s, buf, he)) {
        Request rq; rq.peer = peer;
        std::string head = buf.substr(0, he);
        size_t eol = head.find("\r\n");
        std::string line = head.substr(0, eol);
        size_t a = line.find(' '), b = line.find(' ', a + 1);
        if (a != std::string::npos && b != std::string::npos) {
            rq.method = line.substr(0, a);
            std::string target = line.substr(a + 1, b - a - 1);
            size_t q = target.find('?');
            rq.path = q == std::string::npos ? target : target.substr(0, q);
            if (q != std::string::npos) rq.query = target.substr(q + 1);
            size_t p = eol;
            while (p != std::string::npos && p < head.size()) {
                p += 2; size_t e = head.find("\r\n", p);
                std::string l = head.substr(p, e == std::string::npos ? std::string::npos : e - p);
                size_t c = l.find(':');
                if (c != std::string::npos) rq.headers[util::lower(l.substr(0, c))] = util::trim(l.substr(c + 1));
                p = e;
            }
            size_t len = 0;
            std::map<std::string, std::string>::iterator cl = rq.headers.find("content-length");
            if (cl != rq.headers.end()) len = (size_t)std::strtoull(cl->second.c_str(), 0, 10);
            if (len > (64u << 20)) { resp.send(413, "application/json", "{\"error\":\"request too large\"}"); }
            else {
                rq.body = buf.substr(he + 4);
                char tmp[8192];
                while (rq.body.size() < len) {
                    int n = (int)::recv(s, tmp, sizeof tmp, 0);
                    if (n <= 0) break;
                    rq.body.append(tmp, (size_t)n);
                }
                if (rq.body.size() > len) rq.body.resize(len);
                try { h(rq, resp); }
                catch (const std::exception& e) { resp.send(500, "application/json", std::string("{\"error\":") + "\"internal error\"}"); (void)e; }
            }
        } else resp.send(400, "text/plain", "bad request");
    }
    resp.finish();
#ifdef _WIN32
    shutdown(s, SD_SEND);
#else
    shutdown(s, SHUT_WR);
#endif
    closeSock(s);
}

void Server::serve(const Handler& h) {
    while (!stopping_) {
        sockaddr_in pa;
#ifdef _WIN32
        int pl = sizeof pa;
#else
        socklen_t pl = sizeof pa;
#endif
        sock_t c = ::accept((sock_t)sock_, (sockaddr*)&pa, &pl);
        if (c == SOCK_INVALID) { if (stopping_) break; continue; }
        std::string peer = inet_ntoa(pa.sin_addr);
        std::thread(handleConn, c, h, peer).detach();
    }
}

// ---------- Client ----------
struct Url { std::string host, path; int port = 80; bool https = false; bool ok = false; };

static Url parseUrl(const std::string& u) {
    Url r; std::string rest;
    if (util::startsWith(u, "http://")) rest = u.substr(7);
    else if (util::startsWith(u, "https://")) { r.https = true; r.port = 443; rest = u.substr(8); }
    else return r;
    size_t sl = rest.find('/');
    std::string hp = sl == std::string::npos ? rest : rest.substr(0, sl);
    r.path = sl == std::string::npos ? "/" : rest.substr(sl);
    size_t c = hp.rfind(':');
    if (c != std::string::npos) { r.port = std::atoi(hp.c_str() + c + 1); hp = hp.substr(0, c); }
    r.host = hp; r.ok = !hp.empty();
    return r;
}

static sock_t connectTo(const std::string& host, int port, int timeoutSec, std::string& err) {
    startup();
    addrinfo hints; std::memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_UNSPEC; hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = 0;
    if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0 || !res) {
        err = "cannot resolve host " + host; return SOCK_INVALID;
    }
    sock_t s = SOCK_INVALID;
    for (addrinfo* p = res; p; p = p->ai_next) {
        s = ::socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (s == SOCK_INVALID) continue;
        if (::connect(s, p->ai_addr, (int)p->ai_addrlen) == 0) break;
        closeSock(s); s = SOCK_INVALID;
    }
    freeaddrinfo(res);
    if (s == SOCK_INVALID) { err = "cannot connect to " + host + ":" + std::to_string(port); return s; }
#ifdef _WIN32
    DWORD tv = (DWORD)timeoutSec * 1000;
#else
    struct timeval tv; tv.tv_sec = timeoutSec; tv.tv_usec = 0;
#endif
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof tv);
    return s;
}

static bool sendAll(sock_t s, const std::string& d) {
    size_t off = 0;
    while (off < d.size()) {
        int w = (int)::send(s, d.data() + off, (int)std::min<size_t>(d.size() - off, 1 << 16), 0);
        if (w <= 0) return false;
        off += (size_t)w;
    }
    return true;
}

bool request(const std::string& url, const ClientOptions& opt,
             const std::function<bool(const ClientResult&)>& onHeaders,
             const std::function<bool(const char*, size_t)>& onData, ClientResult& res, std::string& err) {
    std::string cur = url;
    for (int hop = 0; hop <= opt.maxRedirects; hop++) {
        Url u = parseUrl(cur);
        if (!u.ok) { err = "invalid URL: " + cur; return false; }
        if (u.https) {
            err = "HTTPS is not available on Windows XP (no modern TLS). Use an http:// mirror, a local HTTP proxy that "
                  "terminates TLS, or download the file elsewhere and import it with `create`";
            return false;
        }
        sock_t s = connectTo(u.host, u.port, opt.timeoutSec, err);
        if (s == SOCK_INVALID) return false;

        std::string rq = opt.method + " " + u.path + " HTTP/1.1\r\nHost: " + u.host + ":" + std::to_string(u.port) +
                         "\r\nUser-Agent: ollama-xp/0.1\r\nAccept: */*\r\nConnection: close\r\n";
        if (opt.rangeStart > 0) rq += "Range: bytes=" + std::to_string(opt.rangeStart) + "-\r\n";
        for (std::map<std::string, std::string>::const_iterator i = opt.headers.begin(); i != opt.headers.end(); ++i)
            rq += i->first + ": " + i->second + "\r\n";
        if (!opt.body.empty() || opt.method == "POST")
            rq += "Content-Type: " + opt.contentType + "\r\nContent-Length: " + std::to_string(opt.body.size()) + "\r\n";
        rq += "\r\n" + opt.body;
        if (!sendAll(s, rq)) { closeSock(s); err = "send failed"; return false; }

        std::string buf; size_t he = 0;
        if (!readHead(s, buf, he)) { closeSock(s); err = "connection closed before response"; return false; }
        std::string head = buf.substr(0, he);
        std::string body0 = buf.substr(he + 4);
        ClientResult r; r.finalUrl = cur;
        r.status = std::atoi(head.c_str() + (head.find(' ') == std::string::npos ? 0 : head.find(' ') + 1));
        bool chunked = false; std::string location; bool hasCL = false;
        size_t p = head.find("\r\n");
        while (p != std::string::npos) {
            p += 2; size_t e = head.find("\r\n", p);
            std::string l = head.substr(p, e == std::string::npos ? std::string::npos : e - p);
            size_t c = l.find(':');
            if (c != std::string::npos) {
                std::string k = util::lower(l.substr(0, c)), v = util::trim(l.substr(c + 1));
                if (k == "content-length") { r.contentLength = std::strtoull(v.c_str(), 0, 10); hasCL = true; }
                else if (k == "transfer-encoding" && util::lower(v).find("chunked") != std::string::npos) chunked = true;
                else if (k == "location") location = v;
                else if (k == "content-range") {   // bytes START-END/TOTAL
                    size_t sp = v.find(' ');
                    r.rangeStart = std::strtoull(v.c_str() + (sp == std::string::npos ? 0 : sp + 1), 0, 10);
                    size_t sl = v.find('/');
                    if (sl != std::string::npos) r.contentLength = std::strtoull(v.c_str() + sl + 1, 0, 10);
                }
            }
            p = e;
        }
        (void)hasCL;
        if ((r.status == 301 || r.status == 302 || r.status == 303 || r.status == 307 || r.status == 308) && !location.empty()) {
            closeSock(s);
            if (util::startsWith(location, "/")) location = "http://" + u.host + ":" + std::to_string(u.port) + location;
            cur = location; continue;
        }
        res = r;
        if (onHeaders && !onHeaders(r)) { closeSock(s); return true; }

        // Body decoding.
        bool keep = true;
        if (!chunked) {
            if (!body0.empty()) keep = onData(body0.data(), body0.size());
            char tmp[1 << 15];
            while (keep) {
                int n = (int)::recv(s, tmp, sizeof tmp, 0);
                if (n < 0) { err = "network timeout or reset"; closeSock(s); return false; }
                if (n == 0) break;
                keep = onData(tmp, (size_t)n);
            }
        } else {
            std::string pend = body0; char tmp[1 << 15];
            for (;;) {
                size_t le;
                while ((le = pend.find("\r\n")) == std::string::npos) {
                    int n = (int)::recv(s, tmp, sizeof tmp, 0);
                    if (n <= 0) { closeSock(s); return keep; }
                    pend.append(tmp, (size_t)n);
                }
                size_t len = (size_t)std::strtoull(pend.c_str(), 0, 16);
                pend.erase(0, le + 2);
                if (len == 0) break;
                while (pend.size() < len + 2) {
                    int n = (int)::recv(s, tmp, sizeof tmp, 0);
                    if (n <= 0) { closeSock(s); return keep; }
                    pend.append(tmp, (size_t)n);
                }
                if (!onData(pend.data(), len)) { keep = false; break; }
                pend.erase(0, len + 2);
            }
        }
        closeSock(s);
        return true;
    }
    err = "too many redirects";
    return false;
}

std::string hostToUrl(const std::string& spec) {
    std::string s = util::trim(spec);
    if (s.empty()) return "http://127.0.0.1:11434";
    if (!util::startsWith(s, "http://") && !util::startsWith(s, "https://")) s = "http://" + s;
    size_t scheme = s.find("://") + 3;
    if (s.find(':', scheme) == std::string::npos) s += ":11434";
    while (!s.empty() && s[s.size() - 1] == '/') s.erase(s.size() - 1);
    return s;
}

std::string urlDecode(const std::string& s) {
    std::string o;
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '%' && i + 2 < s.size()) { o += (char)std::strtol(s.substr(i + 1, 2).c_str(), 0, 16); i += 2; }
        else if (s[i] == '+') o += ' ';
        else o += s[i];
    }
    return o;
}

}  // namespace http
