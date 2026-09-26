// Search parameters exposed as UCI spin options, so features can be toggled and SPSA-tuned
// without rebuilding. TUNE(name, default, min, max) defines `int name` and registers it.
// Feature switches are TUNE(UseX, 0, 0, 1): default 0 keeps the pre-experiment behaviour.
#pragma once
#include <string>
#include <vector>

struct Tunable {
    const char* name;
    int* v;
    int def, lo, hi;
};
std::vector<Tunable>& tunables();
struct TuneReg {
    TuneReg(const char* n, int* v, int d, int lo, int hi) { tunables().push_back({n, v, d, lo, hi}); }
};
#define TUNE(name, def, lo, hi) \
    int name = def;             \
    static TuneReg reg_##name(#name, &name, def, lo, hi)
bool set_tunable(const std::string& name, int value);  // false if no such tunable
