#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "../bars/dollar_bar.hpp"

// ---------------------------------------------------------------------------
// Build a daily dollar-volume profile from aggTrades parquet files,
// then compute per-day thresholds for dynamic dollar bars.
//
// Usage:
//   DollarProfile profile;
//   build_profile(file_paths, profile);
//   DailyThresholdMap thresholds = compute_thresholds(
//       profile, bars_per_day, lookback_days, discard_months);
// ---------------------------------------------------------------------------

// Maps integer day index (timestamp_ms / 86400000) to total dollar volume.
using DollarProfile = std::map<int64_t, double>;

// Scan a list of aggTrades parquet files and accumulate daily dollar volumes.
// This is the "profile build" first pass — does NOT build bars.
void build_profile(const std::vector<std::string>& file_paths,
                   DollarProfile& profile);

// From the profile, compute a per-day threshold using a rolling SMA.
//
// For each day D with data:
//   - Skip if D falls within the first `discard_months` of data.
//   - threshold[D] = mean(daily_vol[D-lookback .. D-1]) / bars_per_day
//
// Returns a DailyThresholdMap suitable for DollarBarBuilder.
DailyThresholdMap compute_thresholds(const DollarProfile& profile,
                                     int bars_per_day,
                                     int lookback_days,
                                     int discard_months = 1);
