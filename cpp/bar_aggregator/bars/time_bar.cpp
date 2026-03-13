#include "time_bar.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
int64_t parse_interval_ms(const std::string& s) {
    if (s.empty()) throw std::invalid_argument("Empty interval string");

    auto ends_with = [&](const char* suffix) -> bool {
        size_t slen = std::strlen(suffix);
        return s.size() >= slen &&
               s.compare(s.size() - slen, slen, suffix) == 0;
    };

    if (ends_with("min")) {
        return std::stoll(s.substr(0, s.size() - 3)) * 60'000LL;
    }
    if (ends_with("ms")) {
        return std::stoll(s.substr(0, s.size() - 2));
    }
    char unit = s.back();
    int64_t v = std::stoll(s.substr(0, s.size() - 1));
    switch (unit) {
        case 's': return v * 1'000LL;
        case 'h': return v * 3'600'000LL;
        case 'd': return v * 86'400'000LL;
        case 'w': return v * 7LL * 86'400'000LL;
        default:
            throw std::invalid_argument("Unknown interval unit in: " + s);
    }
}

// ---------------------------------------------------------------------------
TimeBarBuilder::TimeBarBuilder(const std::string& interval, bool include_advanced)
    : interval_ms_(parse_interval_ms(interval))
    , include_advanced_(include_advanced) {}

// ---------------------------------------------------------------------------
void TimeBarBuilder::reset_bar(int64_t bar_start, bool is_bt) {
    active_       = true;
    bar_start_    = bar_start;
    bar_end_      = bar_start;
    open_ = high_ = low_ = close_ = 0.0;
    volume_ = buy_vol_ = vwap_sum_ = dollar_vol_ = 0.0;
    tick_count_   = 0;
    is_bookticker_ = is_bt;
    adv_.reset();
    mid_open_ = mid_high_ = mid_low_ = mid_close_ = 0.0;
    spread_max_ = 0.0;
    spread_stats_.reset();
}

// ---------------------------------------------------------------------------
BarRecord TimeBarBuilder::finalize_bar() const {
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
    r.is_bookticker = is_bookticker_;

    if (include_advanced_ && !is_bookticker_) {
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
        double price_range   = high_ - low_;
        r.spread_proxy       = (r.vwap > 0.0 && price_range > 0.0)
                                   ? price_range / r.vwap
                                   : 0.0;
        r.skewness           = adv_.price_stats.skewness();
        r.kurtosis           = adv_.price_stats.kurtosis();
        r.max_trade_volume   = adv_.max_trade_vol;
        r.max_trade_ratio    = volume_ > 0.0
                                   ? adv_.max_trade_vol / volume_
                                   : 0.0;
        r.tick_interval_mean = adv_.interval_stats.mean();
        r.tick_interval_std  = adv_.interval_stats.std_dev();
        double oc_diff       = std::abs(close_ - open_);
        r.path_efficiency    = adv_.price_path > 0.0
                                   ? oc_diff / adv_.price_path
                                   : 1.0;
        r.impact_density     = dollar_vol_ > 0.0
                                   ? oc_diff / dollar_vol_
                                   : 0.0;
    }

    if (is_bookticker_) {
        r.mid_open    = mid_open_;
        r.mid_high    = mid_high_;
        r.mid_low     = mid_low_;
        r.mid_close   = mid_close_;
        r.spread_mean = spread_stats_.mean();
        r.spread_std  = spread_stats_.std_dev();
        r.spread_max  = spread_max_;
    }
    return r;
}

// ---------------------------------------------------------------------------
std::vector<BarRecord> TimeBarBuilder::add_tick(const AggTick& tick) {
    int64_t bar_start = (tick.timestamp_ms / interval_ms_) * interval_ms_;

    std::vector<BarRecord> completed;
    if (active_ && bar_start != bar_start_) {
        completed.push_back(finalize_bar());
        reset_bar(bar_start, false);
    } else if (!active_) {
        reset_bar(bar_start, false);
    }

    bar_end_ = tick.timestamp_ms;

    if (tick_count_ == 0) {
        open_ = high_ = low_ = close_ = tick.price;
    } else {
        if (tick.price > high_) high_ = tick.price;
        if (tick.price < low_)  low_  = tick.price;
        close_ = tick.price;
    }

    volume_     += tick.quantity;
    dollar_vol_ += tick.price * tick.quantity;
    vwap_sum_   += tick.price * tick.quantity;
    if (!tick.is_buyer_maker) buy_vol_ += tick.quantity; // taker buy
    ++tick_count_;

    if (include_advanced_) adv_.push_tick(tick.timestamp_ms, tick.price, tick.quantity);

    return completed;
}

// ---------------------------------------------------------------------------
std::vector<BarRecord> TimeBarBuilder::add_quote(const BookTick& quote) {
    int64_t bar_start = (quote.timestamp_ms / interval_ms_) * interval_ms_;
    double  mid    = (quote.best_bid_price + quote.best_ask_price) * 0.5;
    double  spread = quote.best_ask_price - quote.best_bid_price;

    std::vector<BarRecord> completed;
    if (active_ && bar_start != bar_start_) {
        completed.push_back(finalize_bar());
        reset_bar(bar_start, true);
    } else if (!active_) {
        reset_bar(bar_start, true);
    }

    bar_end_ = quote.timestamp_ms;

    if (tick_count_ == 0) {
        open_ = high_ = low_ = close_ = mid;
        mid_open_ = mid_high_ = mid_low_ = mid_close_ = mid;
    } else {
        if (mid > mid_high_) mid_high_ = mid;
        if (mid < mid_low_)  mid_low_  = mid;
        mid_close_ = mid;
        if (mid > high_) high_ = mid;
        if (mid < low_)  low_  = mid;
        close_ = mid;
    }

    ++tick_count_;
    spread_stats_.push(spread);
    if (spread > spread_max_) spread_max_ = spread;

    return completed;
}

// ---------------------------------------------------------------------------
std::optional<BarRecord> TimeBarBuilder::flush() {
    if (!active_ || tick_count_ == 0) return std::nullopt;
    return finalize_bar();
}
