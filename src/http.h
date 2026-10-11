// Tiny blocking HTTP/1.1 server and client over Winsock (XP: ws2_32) or BSD sockets.
// Plain HTTP only: Windows XP has no usable modern TLS stack.
#pragma once
#include <stdint.h>
#include <functional>
#include <map>
#include <string>

namespace http {

bool startup();   // WSAStartup on Windows, no-op elsewhere

struct Request {
    std::string method, path, query, body;
    std::map<std::string, std::string> headers;   // lower-case keys
    std::string peer;
};

class Response {
public:
    explicit Response(intptr_t sock) : sock_(sock), headSent_(false), chunked_(false), failed_(false) {}
    // Starts a streamed (chunked) response. Use write() then finish().
    bool startStream(int status, const std::string& contentType);
    bool write(const std::string& data);   // false once the client has gone away
    void finish();
    // One-shot response with Content-Length.
    void send(int status, const std::string& contentType, const std::string& body);
    bool failed() const { return failed_; }
private:
    bool sendRaw(const char* p, size_t n);
    intptr_t sock_; bool headSent_, chunked_, failed_;
};

typedef std::function<void(const Request&, Response&)> Handler;

class Server {
public:
    Server();
    ~Server();
    bool listen(const std::string& host, int port, std::string& err);
    void serve(const Handler& h);   // blocks; thread per connection
    void stop();
private:
    intptr_t sock_;
    volatile bool stopping_;
};

struct ClientOptions {
    std::string method = "GET";
    std::string body;
    std::string contentType = "application/json";
    std::map<std::string, std::string> headers;
    uint64_t rangeStart = 0;          // send Range: bytes=N- when > 0
    int maxRedirects = 8;
    int timeoutSec = 60;
};

struct ClientResult {
    int status = 0;
    uint64_t contentLength = 0;       // 0 if unknown
    uint64_t rangeStart = 0;          // offset the body starts at (206 responses)
    std::string finalUrl;
};

// Streams the response body to onData (return false to abort). Returns false on transport errors.
bool request(const std::string& url, const ClientOptions& opt,
             const std::function<bool(const ClientResult&)>& onHeaders,
             const std::function<bool(const char*, size_t)>& onData, ClientResult& result, std::string& err);

// "host:port" / "http://host:port" -> http base URL. Default port 11434.
std::string hostToUrl(const std::string& hostSpec);
std::string urlDecode(const std::string& s);

}  // namespace http
