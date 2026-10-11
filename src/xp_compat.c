/* Windows XP compatibility shim for the 32-bit MinGW build.
 *
 * libwinpthread (pulled in statically by std::thread / std::mutex) imports GetTickCount64, which exists
 * only on Vista and newer; on XP the loader would refuse to start the program. We define the import
 * pointer ourselves so the linker never pulls kernel32's version, and implement it on top of
 * GetTickCount() with wrap-around tracking. */
#if defined(_WIN32) && !defined(_WIN64)
#include <windows.h>

static volatile LONG g_lock = 0;
static DWORD g_last = 0;
static unsigned long long g_high = 0;

static ULONGLONG WINAPI xp_GetTickCount64(void) {
    DWORD now = GetTickCount();
    ULONGLONG r;
    while (InterlockedCompareExchange(&g_lock, 1, 0) != 0) Sleep(0);
    if (now < g_last) g_high += 0x100000000ULL;   /* 49.7 day wrap */
    g_last = now;
    r = g_high + now;
    InterlockedExchange(&g_lock, 0);
    return r;
}

/* Satisfies the undefined symbol `__imp__GetTickCount64@0` referenced by libwinpthread. */
void* xp_imp_GetTickCount64 __asm__("__imp__GetTickCount64@0") = (void*)xp_GetTickCount64;
#endif
