// Small portable helpers: strings, files, hashing, time, console.
#pragma once
#include <stdint.h>
#include <functional>
#include <string>
#include <vector>

namespace util {

std::string trim(const std::string& s);
std::string lower(const std::string& s);
bool startsWith(const std::string& s, const std::string& p);
bool endsWith(const std::string& s, const std::string& p);
std::vector<std::string> split(const std::string& s, char sep);
std::string replaceAll(std::string s, const std::string& from, const std::string& to);

// Ollama-style human sizes: "4.7 GB", "397 MB".
std::string humanBytes(uint64_t n);
// "2 hours ago", "About a minute ago".
std::string humanAgo(int64_t unixSeconds);
// "4 minutes from now" for keep-alive expiry.
std::string humanFromNow(int64_t unixSeconds);
std::string iso8601(int64_t unixSeconds);
int64_t nowUnix();
// Parses "YYYY-MM-DDTHH:MM:SSZ"; returns 0 on failure.
int64_t parseIso8601(const std::string& s);
int64_t nowMillis();

std::string getEnv(const char* name, const std::string& def = "");
std::string homeDir();

// Filesystem (UTF-8 paths).
bool fileExists(const std::string& path);
bool dirExists(const std::string& path);
bool makeDirs(const std::string& path);
bool removeFile(const std::string& path);
bool removeEmptyDirs(const std::string& root, const std::string& stop);
bool copyFile(const std::string& from, const std::string& to);
bool moveFile(const std::string& from, const std::string& to);
int64_t fileSize(const std::string& path);
bool readFile(const std::string& path, std::string& out);
bool writeFile(const std::string& path, const std::string& data);
// Recursively lists regular files below root (full paths, '/' separators).
std::vector<std::string> listFiles(const std::string& root);
std::string joinPath(const std::string& a, const std::string& b);

// SHA-256 (lowercase hex). The file variant streams and reports progress.
std::string sha256Hex(const std::string& data);
std::string sha256File(const std::string& path, const std::function<void(uint64_t done, uint64_t total)>& progress = nullptr);

// Console helpers.
bool stdinIsTty();
bool stdoutIsTty();
void setupConsoleUtf8();
// Writes UTF-8 text to stdout (uses WriteConsoleW on a Windows console so Unicode works on XP).
void consoleWrite(const std::string& text);
// Reads one line from stdin with prompt; returns false on EOF.
bool readLine(const std::string& prompt, std::string& line);

}  // namespace util
