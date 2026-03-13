#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "bar_state.hpp"
#include "../features/microstructure.hpp"

// ---------------------------------------------------------------------------
// Time bar builder.
//
// Bars are floor-aligned to the interval:
//   bar_start = floor(tick_ts / interval_ms) * interval_ms
//
// Supports both aggTrades and bookTicker ticks.
// State persists across row groups and across monthly file boundaries.
// ---------------------------------------------------------------------------

class TimeBarBuilder {
public:
    // Parse interval string ("1min", "5min", "1h", "4h", "1d", etc.)
    // and construct builder. include_advanced only applies to aggTrades.
    explicit TimeBarBuilder(const std::string& interval,
                            bool include_advanced = true);

    // Add one aggTrades tick. Returns newly completed bars (0 or more).
    std::vector<BarRecord> add_tick(const AggTick& tick);

    // Add one bookTicker quote. Returns newly completed bars (0 or more).
    std::vector<BarRecord> add_quote(const BookTick& quote);

    // Flush the incomplete bar (call at end of all processing).
    // Returns the incomplete bar if at least one tick was seen.
    std::optional<BarRecord> flush();

private:
    int64_t interval_ms_;
    bool    include_advanced_;

    // Current incomplete bar state
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

    // bookTicker state
    bool           is_bookticker_ = false;
    double         mid_open_      = 0.0;
    double         mid_high_      = 0.0;
    double         mid_low_       = 0.0;
    double         mid_close_     = 0.0;
    double         spread_max_    = 0.0;
    OnlineVariance spread_stats_;

    void      reset_bar(int64_t bar_start, bool is_bt);
    BarRecord finalize_bar() const;
};

// Parse interval string to milliseconds.
int64_t parse_interval_ms(const std::string& interval);
