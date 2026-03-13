#include "parquet_writer.hpp"

#include <filesystem>
#include <stdexcept>
#include <vector>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>
#include <parquet/properties.h>

namespace io {

namespace {

// Build the Arrow schema for aggTrades bars.
std::shared_ptr<arrow::Schema> aggtrades_schema(bool advanced) {
    arrow::FieldVector fields = {
        arrow::field("start_time_ms", arrow::int64()),
        arrow::field("end_time_ms",   arrow::int64()),
        arrow::field("open",          arrow::float64()),
        arrow::field("high",          arrow::float64()),
        arrow::field("low",           arrow::float64()),
        arrow::field("close",         arrow::float64()),
        arrow::field("volume",        arrow::float64()),
        arrow::field("buy_volume",    arrow::float64()),
        arrow::field("sell_volume",   arrow::float64()),
        arrow::field("vwap",          arrow::float64()),
        arrow::field("tick_count",    arrow::int64()),
        arrow::field("dollar_volume", arrow::float64()),
    };
    if (advanced) {
        fields.push_back(arrow::field("price_std",          arrow::float64()));
        fields.push_back(arrow::field("volume_std",         arrow::float64()));
        fields.push_back(arrow::field("up_move_ratio",      arrow::float64()));
        fields.push_back(arrow::field("down_move_ratio",    arrow::float64()));
        fields.push_back(arrow::field("reversals",          arrow::int64()));
        fields.push_back(arrow::field("buy_sell_imbalance", arrow::float64()));
        fields.push_back(arrow::field("spread_proxy",       arrow::float64()));
        fields.push_back(arrow::field("skewness",           arrow::float64()));
        fields.push_back(arrow::field("kurtosis",           arrow::float64()));
        fields.push_back(arrow::field("max_trade_volume",   arrow::float64()));
        fields.push_back(arrow::field("max_trade_ratio",    arrow::float64()));
        fields.push_back(arrow::field("tick_interval_mean", arrow::float64()));
        fields.push_back(arrow::field("tick_interval_std",  arrow::float64()));
        fields.push_back(arrow::field("path_efficiency",    arrow::float64()));
        fields.push_back(arrow::field("impact_density",     arrow::float64()));
    }
    return arrow::schema(fields);
}

// Build the Arrow schema for bookTicker time bars.
std::shared_ptr<arrow::Schema> bookticker_schema() {
    return arrow::schema({
        arrow::field("start_time_ms", arrow::int64()),
        arrow::field("end_time_ms",   arrow::int64()),
        arrow::field("open",          arrow::float64()),
        arrow::field("high",          arrow::float64()),
        arrow::field("low",           arrow::float64()),
        arrow::field("close",         arrow::float64()),
        arrow::field("tick_count",    arrow::int64()),
        arrow::field("mid_open",      arrow::float64()),
        arrow::field("mid_high",      arrow::float64()),
        arrow::field("mid_low",       arrow::float64()),
        arrow::field("mid_close",     arrow::float64()),
        arrow::field("spread_mean",   arrow::float64()),
        arrow::field("spread_std",    arrow::float64()),
        arrow::field("spread_max",    arrow::float64()),
    });
}

// Append a double value to a DoubleBuilder.
void append(arrow::DoubleBuilder& b, double v) {
    auto s = b.Append(v);
    if (!s.ok()) throw std::runtime_error("DoubleBuilder::Append failed");
}

// Append an int64 value to an Int64Builder.
void append(arrow::Int64Builder& b, int64_t v) {
    auto s = b.Append(v);
    if (!s.ok()) throw std::runtime_error("Int64Builder::Append failed");
}

template <typename B>
std::shared_ptr<arrow::Array> finish(B& b) {
    std::shared_ptr<arrow::Array> arr;
    auto s = b.Finish(&arr);
    if (!s.ok()) throw std::runtime_error("Builder::Finish failed");
    return arr;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
void write_bars(const std::string& path,
                const std::vector<BarRecord>& bars,
                SchemaType schema_type,
                bool include_advanced) {
    if (bars.empty()) return;

    // Ensure parent directory exists.
    std::filesystem::create_directories(
        std::filesystem::path(path).parent_path());

    if (schema_type == SchemaType::BOOKTICKER) {
        // --- bookTicker schema ---
        auto schema = bookticker_schema();

        arrow::Int64Builder  start_b, end_b, tick_b;
        arrow::DoubleBuilder open_b, high_b, low_b, close_b;
        arrow::DoubleBuilder mid_open_b, mid_high_b, mid_low_b, mid_close_b;
        arrow::DoubleBuilder spread_mean_b, spread_std_b, spread_max_b;

        for (const auto& r : bars) {
            append(start_b, r.start_time_ms);
            append(end_b,   r.end_time_ms);
            append(open_b,  r.open);
            append(high_b,  r.high);
            append(low_b,   r.low);
            append(close_b, r.close);
            append(tick_b,  r.tick_count);
            append(mid_open_b,    r.mid_open);
            append(mid_high_b,    r.mid_high);
            append(mid_low_b,     r.mid_low);
            append(mid_close_b,   r.mid_close);
            append(spread_mean_b, r.spread_mean);
            append(spread_std_b,  r.spread_std);
            append(spread_max_b,  r.spread_max);
        }

        auto table = arrow::Table::Make(schema, {
            finish(start_b), finish(end_b),
            finish(open_b),  finish(high_b), finish(low_b), finish(close_b),
            finish(tick_b),
            finish(mid_open_b), finish(mid_high_b),
            finish(mid_low_b),  finish(mid_close_b),
            finish(spread_mean_b), finish(spread_std_b), finish(spread_max_b),
        });

        auto out_result = arrow::io::FileOutputStream::Open(path);
        if (!out_result.ok()) {
            throw std::runtime_error("Cannot open output file: " + path);
        }
        auto outfile = out_result.ValueUnsafe();

        auto props = parquet::WriterProperties::Builder()
                         .compression(parquet::Compression::ZSTD)
                         ->build();
        auto s = parquet::arrow::WriteTable(
            *table, arrow::default_memory_pool(), outfile,
            static_cast<int64_t>(bars.size()), props);
        if (!s.ok()) {
            throw std::runtime_error("WriteTable failed for " + path + ": " + s.ToString());
        }
        return;
    }

    // --- aggTrades schema ---
    auto schema = aggtrades_schema(include_advanced);

    arrow::Int64Builder  start_b, end_b, tick_b;
    arrow::DoubleBuilder open_b, high_b, low_b, close_b;
    arrow::DoubleBuilder vol_b, bvol_b, svol_b, vwap_b, dvol_b;

    // Advanced builders (only allocated if needed)
    arrow::DoubleBuilder pstd_b, vstd_b, up_b, dn_b, imbal_b, spread_b;
    arrow::DoubleBuilder skew_b, kurt_b, maxvol_b, maxrat_b, imean_b, istd_b;
    arrow::DoubleBuilder peff_b, dens_b;
    arrow::Int64Builder  rev_b;

    for (const auto& r : bars) {
        append(start_b, r.start_time_ms);
        append(end_b,   r.end_time_ms);
        append(open_b,  r.open);
        append(high_b,  r.high);
        append(low_b,   r.low);
        append(close_b, r.close);
        append(vol_b,   r.volume);
        append(bvol_b,  r.buy_volume);
        append(svol_b,  r.sell_volume);
        append(vwap_b,  r.vwap);
        append(tick_b,  r.tick_count);
        append(dvol_b,  r.dollar_volume);

        if (include_advanced) {
            append(pstd_b,   r.price_std);
            append(vstd_b,   r.volume_std);
            append(up_b,     r.up_move_ratio);
            append(dn_b,     r.down_move_ratio);
            append(rev_b,    r.reversals);
            append(imbal_b,  r.buy_sell_imbalance);
            append(spread_b, r.spread_proxy);
            append(skew_b,   r.skewness);
            append(kurt_b,   r.kurtosis);
            append(maxvol_b, r.max_trade_volume);
            append(maxrat_b, r.max_trade_ratio);
            append(imean_b,  r.tick_interval_mean);
            append(istd_b,   r.tick_interval_std);
            append(peff_b,   r.path_efficiency);
            append(dens_b,   r.impact_density);
        }
    }

    arrow::ArrayVector arrays = {
        finish(start_b), finish(end_b),
        finish(open_b),  finish(high_b), finish(low_b), finish(close_b),
        finish(vol_b),   finish(bvol_b), finish(svol_b),
        finish(vwap_b),  finish(tick_b), finish(dvol_b),
    };
    if (include_advanced) {
        arrays.push_back(finish(pstd_b));
        arrays.push_back(finish(vstd_b));
        arrays.push_back(finish(up_b));
        arrays.push_back(finish(dn_b));
        arrays.push_back(finish(rev_b));
        arrays.push_back(finish(imbal_b));
        arrays.push_back(finish(spread_b));
        arrays.push_back(finish(skew_b));
        arrays.push_back(finish(kurt_b));
        arrays.push_back(finish(maxvol_b));
        arrays.push_back(finish(maxrat_b));
        arrays.push_back(finish(imean_b));
        arrays.push_back(finish(istd_b));
        arrays.push_back(finish(peff_b));
        arrays.push_back(finish(dens_b));
    }

    auto table = arrow::Table::Make(schema, arrays);

    auto out_result = arrow::io::FileOutputStream::Open(path);
    if (!out_result.ok()) {
        throw std::runtime_error("Cannot open output file: " + path);
    }
    auto outfile = out_result.ValueUnsafe();

    auto props = parquet::WriterProperties::Builder()
                     .compression(parquet::Compression::ZSTD)
                     ->build();
    auto s = parquet::arrow::WriteTable(
        *table, arrow::default_memory_pool(), outfile,
        static_cast<int64_t>(bars.size()), props);
    if (!s.ok()) {
        throw std::runtime_error("WriteTable failed for " + path + ": " + s.ToString());
    }
}

} // namespace io
