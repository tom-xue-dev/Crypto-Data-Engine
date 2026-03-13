/**
 * bar_aggregator — C++ streaming bar aggregation binary.
 *
 * Usage:
 *   bar_aggregator \
 *     --input-dir  <path>           Root directory for input parquet files
 *     --symbol     <BTCUSDT>        Single symbol to process
 *     --data-type  <aggtrades|bookticker>
 *     --start      <YYYY-MM>        First month to process
 *     --end        <YYYY-MM>        Last month to process (inclusive)
 *     --bar-type   <time_bar|volume_bar|dollar_bar>
 *     --threshold  <number|auto>    Bar threshold; "auto" = dynamic dollar bar
 *     --output-dir <path>           Root directory for output parquet files
 *     [--include-advanced]          Include advanced microstructure features
 *     [--bars-per-day  50]          Target bars/day for auto threshold
 *     [--lookback-days 10]          Rolling lookback for auto threshold
 *
 * Output layout:
 *   {output_dir}/{symbol}/{YYYY-MM}.parquet
 *
 * Progress is written to stderr:
 *   [INFO] BTCUSDT 2024-03: 1523 bars written
 */

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "bars/bar_state.hpp"
#include "bars/time_bar.hpp"
#include "bars/volume_bar.hpp"
#include "bars/dollar_bar.hpp"
#include "profile/dollar_profile.hpp"
#include "io/parquet_reader.hpp"
#include "io/parquet_writer.hpp"

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
struct Config {
    std::string input_dir;
    std::string symbol;
    std::string data_type   = "aggtrades";   // aggtrades | bookticker
    std::string start_month;                  // YYYY-MM
    std::string end_month;
    std::string bar_type    = "dollar_bar";   // time_bar | volume_bar | dollar_bar
    std::string threshold   = "auto";         // numeric string or "auto"
    std::string output_dir;
    bool        include_advanced = true;
    int         bars_per_day     = 50;
    int         lookback_days    = 10;
    int         discard_months   = 1;
};

// ---------------------------------------------------------------------------
// CLI parsing
// ---------------------------------------------------------------------------
static Config parse_args(int argc, char* argv[]) {
    Config cfg;
    auto require_next = [&](int& i, const char* flag) -> std::string {
        if (i + 1 >= argc) {
            throw std::invalid_argument(std::string("Missing value for ") + flag);
        }
        return std::string(argv[++i]);
    };

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if      (arg == "--input-dir")       cfg.input_dir        = require_next(i, arg.c_str());
        else if (arg == "--symbol")          cfg.symbol           = require_next(i, arg.c_str());
        else if (arg == "--data-type")       cfg.data_type        = require_next(i, arg.c_str());
        else if (arg == "--start")           cfg.start_month      = require_next(i, arg.c_str());
        else if (arg == "--end")             cfg.end_month        = require_next(i, arg.c_str());
        else if (arg == "--bar-type")        cfg.bar_type         = require_next(i, arg.c_str());
        else if (arg == "--threshold")       cfg.threshold        = require_next(i, arg.c_str());
        else if (arg == "--output-dir")      cfg.output_dir       = require_next(i, arg.c_str());
        else if (arg == "--include-advanced") cfg.include_advanced = true;
        else if (arg == "--no-advanced")     cfg.include_advanced = false;
        else if (arg == "--bars-per-day")    cfg.bars_per_day     = std::stoi(require_next(i, arg.c_str()));
        else if (arg == "--lookback-days")   cfg.lookback_days    = std::stoi(require_next(i, arg.c_str()));
        else if (arg == "--discard-months")  cfg.discard_months   = std::stoi(require_next(i, arg.c_str()));
        else {
            throw std::invalid_argument("Unknown argument: " + arg);
        }
    }

    // Validate required fields
    if (cfg.input_dir.empty())   throw std::invalid_argument("--input-dir is required");
    if (cfg.symbol.empty())      throw std::invalid_argument("--symbol is required");
    if (cfg.start_month.empty()) throw std::invalid_argument("--start is required");
    if (cfg.end_month.empty())   throw std::invalid_argument("--end is required");
    if (cfg.output_dir.empty())  throw std::invalid_argument("--output-dir is required");

    return cfg;
}

// ---------------------------------------------------------------------------
// Month enumeration (YYYY-MM strings in [start, end])
// ---------------------------------------------------------------------------
static std::vector<std::string> enumerate_months(const std::string& start,
                                                  const std::string& end) {
    // Parse YYYY-MM
    auto parse = [](const std::string& s) -> std::pair<int,int> {
        return {std::stoi(s.substr(0, 4)), std::stoi(s.substr(5, 2))};
    };
    auto [sy, sm] = parse(start);
    auto [ey, em] = parse(end);

    std::vector<std::string> months;
    int y = sy, m = sm;
    while (y < ey || (y == ey && m <= em)) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "%04d-%02d", y, m);
        months.emplace_back(buf);
        if (++m > 12) { m = 1; ++y; }
    }
    return months;
}

// ---------------------------------------------------------------------------
// Build file path for a given month
// ---------------------------------------------------------------------------
static std::string aggtrades_path(const Config& cfg, const std::string& month) {
    // {input_dir}/{symbol}/{symbol}-aggTrades-{YYYY-MM}.parquet
    return (fs::path(cfg.input_dir) / cfg.symbol /
            (cfg.symbol + "-aggTrades-" + month + ".parquet")).string();
}

static std::string bookticker_path(const Config& cfg, const std::string& month) {
    // {input_dir}/bookTicker/{symbol}/{symbol}-bookTicker-{YYYY-MM}.parquet
    return (fs::path(cfg.input_dir) / "bookTicker" / cfg.symbol /
            (cfg.symbol + "-bookTicker-" + month + ".parquet")).string();
}

static std::string output_path(const Config& cfg, const std::string& month) {
    return (fs::path(cfg.output_dir) / cfg.symbol / (month + ".parquet")).string();
}

// ---------------------------------------------------------------------------
// aggTrades processing
// ---------------------------------------------------------------------------
static void process_aggtrades(const Config& cfg) {
    std::vector<std::string> months = enumerate_months(cfg.start_month, cfg.end_month);

    // --- Build dollar profile first (if auto threshold) ---
    DollarProfile     profile;
    DailyThresholdMap thresholds;

    if (cfg.bar_type == "dollar_bar" && cfg.threshold == "auto") {
        std::cerr << "[INFO] " << cfg.symbol << ": building dollar volume profile...\n";
        std::vector<std::string> all_files;
        for (const auto& m : months) {
            std::string p = aggtrades_path(cfg, m);
            if (fs::exists(p)) all_files.push_back(p);
        }
        build_profile(all_files, profile);
        thresholds = compute_thresholds(profile,
                                        cfg.bars_per_day,
                                        cfg.lookback_days,
                                        cfg.discard_months);
        std::cerr << "[INFO] " << cfg.symbol << ": profile built, "
                  << thresholds.size() << " threshold days\n";
    }

    // --- Create bar builder ---
    std::unique_ptr<TimeBarBuilder>   time_builder;
    std::unique_ptr<VolumeBarBuilder> vol_builder;
    std::unique_ptr<DollarBarBuilder> dollar_builder;

    if (cfg.bar_type == "time_bar") {
        time_builder = std::make_unique<TimeBarBuilder>(
            cfg.threshold, cfg.include_advanced);
    } else if (cfg.bar_type == "volume_bar") {
        double thr = std::stod(cfg.threshold);
        vol_builder = std::make_unique<VolumeBarBuilder>(thr, cfg.include_advanced);
    } else if (cfg.bar_type == "dollar_bar") {
        if (cfg.threshold == "auto") {
            dollar_builder = std::make_unique<DollarBarBuilder>(
                &thresholds, cfg.include_advanced);
        } else {
            double thr = std::stod(cfg.threshold);
            dollar_builder = std::make_unique<DollarBarBuilder>(thr, cfg.include_advanced);
        }
    } else {
        throw std::invalid_argument("Unknown bar-type: " + cfg.bar_type);
    }

    // --- Process month by month ---
    for (const auto& month : months) {
        std::string in_path = aggtrades_path(cfg, month);
        if (!fs::exists(in_path)) {
            std::cerr << "[WARN] " << cfg.symbol << " " << month
                      << ": file not found, skipping\n";
            continue;
        }

        std::vector<BarRecord> month_bars;

        auto handle_bars = [&](std::vector<BarRecord> completed) {
            for (auto& b : completed) month_bars.push_back(std::move(b));
        };

        io::read_aggtrades(in_path, [&](const std::vector<AggTick>& batch) {
            for (const auto& tick : batch) {
                if (time_builder)   handle_bars(time_builder->add_tick(tick));
                if (vol_builder)    handle_bars(vol_builder->add_tick(tick));
                if (dollar_builder) handle_bars(dollar_builder->add_tick(tick));
            }
        });

        if (!month_bars.empty()) {
            std::string out = output_path(cfg, month);
            io::write_bars(out, month_bars,
                           io::SchemaType::AGGTRADES, cfg.include_advanced);
        }

        std::cerr << "[INFO] " << cfg.symbol << " " << month << ": "
                  << month_bars.size() << " bars written\n";
    }

    // --- Flush remaining incomplete bar ---
    std::optional<BarRecord> final_bar;
    if (time_builder)   final_bar = time_builder->flush();
    if (vol_builder)    final_bar = vol_builder->flush();
    if (dollar_builder) final_bar = dollar_builder->flush();

    if (final_bar.has_value()) {
        // Append to the last month's file by re-reading — or write separately.
        // Simplest: write a "_tail" file for the incomplete bar.
        // In practice the incomplete bar is usually small; users can ignore it.
        std::string tail_path = (fs::path(cfg.output_dir) / cfg.symbol /
                                 "_incomplete_bar.parquet").string();
        io::write_bars(tail_path, {*final_bar},
                       io::SchemaType::AGGTRADES, cfg.include_advanced);
        std::cerr << "[INFO] " << cfg.symbol
                  << ": 1 incomplete bar written to _incomplete_bar.parquet\n";
    }
}

// ---------------------------------------------------------------------------
// bookTicker processing (time bar only)
// ---------------------------------------------------------------------------
static void process_bookticker(const Config& cfg) {
    if (cfg.bar_type != "time_bar") {
        throw std::invalid_argument(
            "bookTicker only supports time_bar (got: " + cfg.bar_type + ")");
    }

    std::vector<std::string> months = enumerate_months(cfg.start_month, cfg.end_month);

    TimeBarBuilder builder(cfg.threshold, /*include_advanced=*/false);

    for (const auto& month : months) {
        std::string in_path = bookticker_path(cfg, month);
        if (!fs::exists(in_path)) {
            std::cerr << "[WARN] " << cfg.symbol << " " << month
                      << ": bookTicker file not found, skipping\n";
            continue;
        }

        std::vector<BarRecord> month_bars;

        io::read_bookticker(in_path, [&](const std::vector<BookTick>& batch) {
            for (const auto& quote : batch) {
                auto completed = builder.add_quote(quote);
                for (auto& b : completed) month_bars.push_back(std::move(b));
            }
        });

        if (!month_bars.empty()) {
            std::string out = output_path(cfg, month);
            io::write_bars(out, month_bars,
                           io::SchemaType::BOOKTICKER, /*include_advanced=*/false);
        }

        std::cerr << "[INFO] " << cfg.symbol << " " << month << ": "
                  << month_bars.size() << " bars written\n";
    }

    auto final_bar = builder.flush();
    if (final_bar.has_value()) {
        std::string tail_path = (fs::path(cfg.output_dir) / cfg.symbol /
                                 "_incomplete_bar.parquet").string();
        io::write_bars(tail_path, {*final_bar},
                       io::SchemaType::BOOKTICKER, /*include_advanced=*/false);
        std::cerr << "[INFO] " << cfg.symbol
                  << ": 1 incomplete bar written to _incomplete_bar.parquet\n";
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    try {
        Config cfg = parse_args(argc, argv);

        std::cerr << "[INFO] bar_aggregator starting\n"
                  << "[INFO]   symbol    : " << cfg.symbol    << "\n"
                  << "[INFO]   data_type : " << cfg.data_type << "\n"
                  << "[INFO]   bar_type  : " << cfg.bar_type  << "\n"
                  << "[INFO]   threshold : " << cfg.threshold << "\n"
                  << "[INFO]   period    : " << cfg.start_month
                                             << " → " << cfg.end_month << "\n"
                  << "[INFO]   advanced  : " << (cfg.include_advanced ? "yes" : "no") << "\n";

        if (cfg.data_type == "aggtrades") {
            process_aggtrades(cfg);
        } else if (cfg.data_type == "bookticker") {
            process_bookticker(cfg);
        } else {
            throw std::invalid_argument("Unknown data-type: " + cfg.data_type);
        }

        std::cerr << "[INFO] " << cfg.symbol << ": done\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }
}
