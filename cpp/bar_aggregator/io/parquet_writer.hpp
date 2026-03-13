#pragma once
#include <string>
#include <vector>

#include "../bars/bar_state.hpp"

namespace io {

enum class SchemaType { AGGTRADES, BOOKTICKER };

// Write a vector of BarRecords to a parquet file.
// Creates parent directories if they don't exist.
// Compression: ZSTD.
void write_bars(const std::string& path,
                const std::vector<BarRecord>& bars,
                SchemaType schema_type,
                bool include_advanced);

} // namespace io
