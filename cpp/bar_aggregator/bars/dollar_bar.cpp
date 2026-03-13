#include "dollar_bar.hpp"

#include <cmath>
#include <stdexcept>

static constexpr int64_t MS_PER_DAY = 86'400'000LL;

DollarBarBuilder::DollarBarBuilder(double fixed_threshold, bool include_advanced)
    : fixed_threshold_(fixed_threshold)
    , include_advanced_(include_advanced) {}

DollarBarBuilder::DollarBarBuilder(const DailyThresholdMap* daily_thresholds,
                                   bool include_advanced)
    : fixed_threshold_(0.0)
    , daily_thresholds_(daily_thresholds)
    , include_advanced_(include_advanced) {}

double DollarBarBuilder::get_threshold(int64_t timestamp_ms) const {
    if (daily_thresholds_ == nullptr) return fixed_threshold_;

    int64_t day_idx = timestamp_ms / MS_PER_DAY;

    // Find the most recent day that has a threshold entry.
    auto it = daily_thresholds_->upper_bound(day_idx);
    if (it == daily_thresholds_->begin()) {
        // No entry at or before this day — use the first entry.
        return daily_thresholds_->begin()->second;
    }
    --it;
    return it->second;
}

void DollarBarBuilder::reset_bar(int64_t ts) {
    active_     = true;
    bar_start_  = ts;
    bar_end_    = ts;
    open_ = high_ = low_ = close_ = 0.0;
    volume_ = buy_vol_ = vwap_sum_ = dollar_vol_ = 0.0;
    tick_count_ = 0;
    adv_.reset();
}

BarRecord DollarBarBuilder::finalize_bar() const {
    BarRecord r;
    r.start_time_ms = bar_start_;
    r.end_time_ms   = bar_end_;
    r.open          = open_;
    r.high          = high_;
    r.low           = low_;
    r.close         = close_;
    r.volume        = volume_;
    r.buy_volume    = buy_vol_;
    r.sell_volume   = volume_ - buy_vol_;
    r.vwap          = volume_ > 0.0 ? vwap_sum_ / volume_ : 0.0;
    r.tick_count    = tick_count_;
    r.dollar_volume = dollar_vol_;

    if (include_advanced_) {
        r.has_advanced       = true;
        r.price_std          = adv_.price_stats.std_dev();
        r.volume_std         = adv_.volume_stats.std_dev();
        r.up_move_ratio      = tick_count_ > 0
                                   ? static_cast<double>(adv_.up_moves) / tick_count_
                                   : 0.0;
        r.down_move_ratio    = tick_count_ > 0
                                   ? static_cast<double>(adv_.down_moves) / tick_count_
                                   : 0.0;
        r.reversals          = adv_.reversals;
        double sell_vol      = volume_ - buy_vol_;
        r.buy_sell_imbalance = volume_ > 0.0
                                   ? (buy_vol_ - sell_vol) / volume_
                                   : 0.0;
        double price_range   = r.high - r.low;
        r.spread_proxy       = (r.vwap > 0.0 && price_range > 0.0)
                                   ? price_range / r.vwap
                                   : 0.0;
        r.skewness           = adv_.price_stats.skewness();
        r.kurtosis           = adv_.price_stats.kurtosis();
        r.max_trade_volume   = adv_.max_trade_vol;
        r.max_trade_ratio    = volume_ > 0.0 ? adv_.max_trade_vol / volume_ : 0.0;
        r.tick_interval_mean = adv_.interval_stats.mean();
        r.tick_interval_std  = adv_.interval_stats.std_dev();
        double oc_diff       = std::abs(close_ - open_);
        r.path_efficiency    = adv_.price_path > 0.0
                                   ? oc_diff / adv_.price_path
                                   : 1.0;
        r.impact_density     = dollar_vol_ > 0.0 ? oc_diff / dollar_vol_ : 0.0;
    }
    return r;
}

std::vector<BarRecord> DollarBarBuilder::add_tick(const AggTick& tick) {
    if (!active_) reset_bar(tick.timestamp_ms);

    std::vector<BarRecord> completed;
    double threshold = get_threshold(tick.timestamp_ms);

    double remaining_dollar = tick.price * tick.quantity;
    double remaining_qty    = tick.quantity;

    while (remaining_dollar > 0.0) {
        double space = threshold - dollar_vol_;
        double used_dollar = remaining_dollar < space ? remaining_dollar : space;
        double used_qty    = remaining_qty * (used_dollar / (tick.price * tick.quantity));
        // Guard against division by zero
        if (tick.price * tick.quantity == 0.0) {
            used_qty = remaining_qty;
        }

        bar_end_ = tick.timestamp_ms;

        if (tick_count_ == 0) {
            open_ = high_ = low_ = close_ = tick.price;
        } else {
            if (tick.price > high_) high_ = tick.price;
            if (tick.price < low_)  low_  = tick.price;
            close_ = tick.price;
        }

        volume_     += used_qty;
        dollar_vol_ += used_dollar;
        vwap_sum_   += tick.price * used_qty;
        if (!tick.is_buyer_maker) buy_vol_ += used_qty;
        ++tick_count_;

        if (include_advanced_) adv_.push_tick(tick.timestamp_ms, tick.price, used_qty);

        remaining_dollar -= used_dollar;
        remaining_qty    -= used_qty;

        if (dollar_vol_ >= threshold) {
            completed.push_back(finalize_bar());
            reset_bar(tick.timestamp_ms);
            // Recalculate threshold for the new bar (day may have changed)
            threshold = get_threshold(tick.timestamp_ms);
        }
    }
    return completed;
}

std::optional<BarRecord> DollarBarBuilder::flush() {
    if (!active_ || tick_count_ == 0) return std::nullopt;
    return finalize_bar();
}
