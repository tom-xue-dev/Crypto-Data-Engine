#pragma once
#include <cstdint>
#include <vector>

// ---------------------------------------------------------------------------
// Tick-level data structs (one row from a parquet file)
// ---------------------------------------------------------------------------

struct AggTick {
    int64_t timestamp_ms;   // milliseconds since epoch
    double  price;
    double  quantity;
    bool    is_buyer_maker; // true = sell-initiated (buyer is the maker)
};

struct BookTick {
    int64_t timestamp_ms;   // transaction_time, ms since epoch
    double  best_bid_price;
    double  best_bid_qty;
    double  best_ask_price;
    double  best_ask_qty;
};

// ---------------------------------------------------------------------------
// Completed bar record
// ---------------------------------------------------------------------------

struct BarRecord {
    // --- Basic (always present) ---
    int64_t start_time_ms = 0;
    int64_t end_time_ms   = 0;
    double  open          = 0.0;
    double  high          = 0.0;
    double  low           = 0.0;
    double  close         = 0.0;
    double  volume        = 0.0;    // total qty
    double  buy_volume    = 0.0;    // taker-buy qty
    double  sell_volume   = 0.0;    // taker-sell qty
    double  vwap          = 0.0;
    int64_t tick_count    = 0;
    double  dollar_volume = 0.0;    // Σ price*qty

    // --- Advanced microstructure (optional) ---
    double  price_std          = 0.0;
    double  volume_std         = 0.0;
    double  up_move_ratio      = 0.0;
    double  down_move_ratio    = 0.0;
    int64_t reversals          = 0;
    double  buy_sell_imbalance = 0.0;
    double  spread_proxy       = 0.0;
    double  skewness           = 0.0;
    double  kurtosis           = 0.0;
    double  max_trade_volume   = 0.0;
    double  max_trade_ratio    = 0.0;
    double  tick_interval_mean = 0.0;
    double  tick_interval_std  = 0.0;
    double  path_efficiency    = 0.0;
    double  impact_density     = 0.0;

    bool has_advanced = false;

    // --- BookTicker-specific (only for bookTicker time bars) ---
    double  mid_open    = 0.0;
    double  mid_high    = 0.0;
    double  mid_low     = 0.0;
    double  mid_close   = 0.0;
    double  spread_mean = 0.0;
    double  spread_std  = 0.0;
    double  spread_max  = 0.0;

    bool is_bookticker = false;
};
