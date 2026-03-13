#include "dollar_profile.hpp"

#include <algorithm>
#include <deque>
#include <stdexcept>
#include <vector>

#include "../io/parquet_reader.hpp"

static constexpr int64_t MS_PER_DAY = 86'400'000LL;

// ---------------------------------------------------------------------------
void build_profile(const std::vector<std::string>& file_paths,
                   DollarProfile& profile) {
    for (const auto& path : file_paths) {
        io::read_aggtrades(path, [&](const std::vector<AggTick>& batch) {
            for (const auto& tick : batch) {
                int64_t day_idx = tick.timestamp_ms / MS_PER_DAY;
                profile[day_idx] += tick.price * tick.quantity;
            }
        });
    }
}

// ---------------------------------------------------------------------------
DailyThresholdMap compute_thresholds(const DollarProfile& profile,
                                     int bars_per_day,
                                     int lookback_days,
                                     int discard_months) {
    if (profile.empty()) return {};
    if (bars_per_day <= 0)  throw std::invalid_argument("bars_per_day must be > 0");
    if (lookback_days <= 0) throw std::invalid_argument("lookback_days must be > 0");

    // Sort days
    std::vector<std::pair<int64_t, double>> days(profile.begin(), profile.end());
    std::sort(days.begin(), days.end());

    // Determine cutoff: skip first `discard_months` months of data.
    // Approximate: discard_months * 30 days.
    int64_t first_day  = days.front().first;
    int64_t cutoff_day = first_day + static_cast<int64_t>(discard_months) * 30LL;

    DailyThresholdMap result;

    // Rolling window of recent daily volumes.
    std::deque<double> window;
    double             window_sum = 0.0;

    for (size_t i = 0; i < days.size(); ++i) {
        int64_t day_idx    = days[i].first;
        double  day_vol    = days[i].second;

        // Compute threshold from the window *before* adding today.
        if (day_idx > cutoff_day && !window.empty()) {
            double mean_vol  = window_sum / static_cast<double>(window.size());
            result[day_idx]  = mean_vol / bars_per_day;
        }

        // Update rolling window.
        window.push_back(day_vol);
        window_sum += day_vol;
        while (static_cast<int>(window.size()) > lookback_days) {
            window_sum -= window.front();
            window.pop_front();
        }
    }

    return result;
}
