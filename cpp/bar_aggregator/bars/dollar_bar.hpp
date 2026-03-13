#pragma once
#include <cstdint>
#include <map>
#include <optional>
#include <vector>

#include "bar_state.hpp"
#include "../features/microstructure.hpp"

// ---------------------------------------------------------------------------
// Dollar bar builder — closes a bar when Σ(price × qty) reaches the threshold.
//
// Supports two threshold modes:
//   Fixed  — constant threshold supplied at construction.
//   Dynamic — per-day threshold looked up from a DailyThresholdMap.
//             Key: integer day index = timestamp_ms / 86400000.
//             Value: threshold for that day.
// ---------------------------------------------------------------------------

using DailyThresholdMap = std::map<int64_t, double>; // day_index → threshold

class DollarBarBuilder {
public:
    // Fixed threshold mode.
    explicit DollarBarBuilder(double fixed_threshold,
                              bool include_advanced = true);

    // Dynamic threshold mode.
    // Requires a populated DailyThresholdMap.
    explicit DollarBarBuilder(const DailyThresholdMap* daily_thresholds,
                              bool include_advanced = true);

    std::vector<BarRecord> add_tick(const AggTick& tick);
    std::optional<BarRecord> flush();

private:
    double                   fixed_threshold_;
    const DailyThresholdMap* daily_thresholds_ = nullptr;
    bool                     include_advanced_;

    bool    active_     = false;
    int64_t bar_start_  = 0;
    int64_t bar_end_    = 0;
    double  open_       = 0.0;
    double  high_       = 0.0;
    double  low_        = 0.0;
    double  close_      = 0.0;
    double  volume_     = 0.0;
    double  buy_vol_    = 0.0;
    double  vwap_sum_   = 0.0;
    int64_t tick_count_ = 0;
    double  dollar_vol_ = 0.0;
    AdvancedAccumulator adv_;

    double get_threshold(int64_t timestamp_ms) const;
    void   reset_bar(int64_t ts);
    BarRecord finalize_bar() const;
};
