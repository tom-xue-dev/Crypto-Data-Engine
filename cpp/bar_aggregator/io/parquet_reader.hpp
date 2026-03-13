#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "../bars/bar_state.hpp"

// ---------------------------------------------------------------------------
// Streaming Parquet reader for aggTrades and bookTicker files.
//
// Reads one row-group at a time to keep memory usage bounded.
// Calls the provided callback for each batch of ticks.
// ---------------------------------------------------------------------------

namespace io {

// Callback types — invoked once per row group.
using AggTickCallback  = std::function<void(const std::vector<AggTick>&)>;
using BookTickCallback = std::function<void(const std::vector<BookTick>&)>;

// Read all row groups from an aggTrades parquet file.
// Calls cb(batch) once per row group.
// Throws std::runtime_error on schema mismatch or read failures.
void read_aggtrades(const std::string& path, const AggTickCallback& cb);

// Read all row groups from a bookTicker parquet file.
// Calls cb(batch) once per row group.
void read_bookticker(const std::string& path, const BookTickCallback& cb);

} // namespace io
