#pragma once
#include <cmath>
#include <cstdint>
#include <limits>

// ---------------------------------------------------------------------------
// Welford's online algorithm for mean, variance, skewness, and kurtosis.
// Single-pass, numerically stable.
// ---------------------------------------------------------------------------
class OnlineStats {
public:
    void push(double x) {
        int64_t n1 = n_;
        ++n_;
        double delta    = x - M1_;
        double delta_n  = delta / n_;
        double delta_n2 = delta_n * delta_n;
        double term1    = delta * delta_n * n1;

        M1_ += delta_n;
        M4_ += term1 * delta_n2 * (static_cast<double>(n_) * n_ - 3 * n_ + 3)
               + 6.0 * delta_n2 * M2_
               - 4.0 * delta_n  * M3_;
        M3_ += term1 * delta_n * static_cast<double>(n_ - 2) - 3.0 * delta_n * M2_;
        M2_ += term1;
    }

    int64_t count()    const { return n_; }
    double  mean()     const { return M1_; }
    double  variance() const { return n_ > 1 ? M2_ / (n_ - 1) : 0.0; }
    double  std_dev()  const { return std::sqrt(variance()); }

    double skewness() const {
        if (n_ < 3 || M2_ == 0.0) return 0.0;
        return std::sqrt(static_cast<double>(n_)) * M3_ / std::pow(M2_, 1.5);
    }

    double kurtosis() const {
        if (n_ < 4 || M2_ == 0.0) return 0.0;
        return (static_cast<double>(n_) * M4_) / (M2_ * M2_) - 3.0;
    }

    void reset() { n_ = 0; M1_ = M2_ = M3_ = M4_ = 0.0; }

private:
    int64_t n_ = 0;
    double  M1_ = 0.0, M2_ = 0.0, M3_ = 0.0, M4_ = 0.0;
};

// ---------------------------------------------------------------------------
// Welford's online algorithm for variance only (lighter than OnlineStats).
// Used for volume_std and spread_std.
// ---------------------------------------------------------------------------
class OnlineVariance {
public:
    void push(double x) {
        ++n_;
        double delta  = x - mean_;
        mean_ += delta / n_;
        double delta2 = x - mean_;
        M2_   += delta * delta2;
    }

    int64_t count()    const { return n_; }
    double  mean()     const { return mean_; }
    double  variance() const { return n_ > 1 ? M2_ / (n_ - 1) : 0.0; }
    double  std_dev()  const { return std::sqrt(variance()); }

    void reset() { n_ = 0; mean_ = M2_ = 0.0; }

private:
    int64_t n_    = 0;
    double  mean_ = 0.0;
    double  M2_   = 0.0;
};

// ---------------------------------------------------------------------------
// Accumulator for all advanced microstructure features within one bar.
// Call push_tick() for each trade; call finalize() to get results.
// ---------------------------------------------------------------------------
struct AdvancedAccumulator {
    OnlineStats   price_stats;
    OnlineVariance volume_stats;

    // Direction tracking
    int64_t up_moves   = 0;
    int64_t down_moves = 0;
    int64_t reversals  = 0;
    int     prev_dir   = 0; // -1, 0, +1

    // Max trade
    double max_trade_vol = 0.0;

    // Tick intervals
    OnlineVariance interval_stats;
    int64_t        prev_ts_ms = -1;

    // Path efficiency numerator
    double price_path = 0.0;
    double first_price = std::numeric_limits<double>::quiet_NaN();
    double last_price  = 0.0;

    void push_tick(int64_t ts_ms, double price, double qty) {
        price_stats.push(price);
        volume_stats.push(qty);

        if (max_trade_vol < qty) max_trade_vol = qty;

        // Direction and reversals
        if (price_stats.count() > 1) {
            int dir = 0;
            if (price > last_price)       dir = +1;
            else if (price < last_price)  dir = -1;

            if (dir == +1)       ++up_moves;
            else if (dir == -1)  ++down_moves;

            if (dir != 0 && prev_dir != 0 && dir != prev_dir) ++reversals;
            if (dir != 0) prev_dir = dir;
        }

        // Tick intervals
        if (prev_ts_ms >= 0) {
            double gap = static_cast<double>(ts_ms - prev_ts_ms);
            if (gap >= 0.0) interval_stats.push(gap);
        }
        prev_ts_ms = ts_ms;

        // Path efficiency: cumulative absolute price changes
        if (std::isnan(first_price)) first_price = price;
        else price_path += std::abs(price - last_price);

        last_price = price;
    }

    void reset() {
        price_stats.reset();
        volume_stats.reset();
        up_moves = down_moves = reversals = 0;
        prev_dir = 0;
        max_trade_vol = 0.0;
        interval_stats.reset();
        prev_ts_ms = -1;
        price_path = 0.0;
        first_price = std::numeric_limits<double>::quiet_NaN();
        last_price = 0.0;
    }
};
