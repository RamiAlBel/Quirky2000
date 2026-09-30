#pragma once
#include "position.h"
#include <string>

namespace book {
bool open(const std::string& path);  // Polyglot .bin; false if missing/empty
bool loaded();
uint64_t key(const Position& pos);   // Polyglot hash
Move probe(const Position& pos, bool best = false);     // weighted-random book move, 0 if none
}
