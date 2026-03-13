#pragma once
#include <cstdint>
#include <optional>
#include <vector>

#include "bar_state.hpp"
#include "../features/microstructure.hpp"

// ---------------------------------------------------------------------------
// Volume bar builder — closes a bar when cumulative traded quantity
// reaches the threshold.  aggTrades only.
// ---------------------------------------------------------------------------

class VolumeBarBuilder {
public:
    explicit VolumeBarBuilder(double threshold, bool include_advanced = true);

    std::vector<BarRecord> add_tick(const AggTick& tick);
    std::optional<BarRecord> flush();

private:
    double  threshold_;
    bool    include_advanced_;

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

    void      reset_bar(int64_t ts);
    BarRecord finalize_bar() const;
};
