#include "parquet_reader.hpp"

#include <stdexcept>
#include <string>
#include <vector>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/type.h>
#include <arrow/type_traits.h>
#include <parquet/arrow/reader.h>

namespace io {

namespace {

// Convert an Arrow timestamp value to milliseconds.
int64_t timestamp_to_ms(int64_t value, arrow::TimeUnit::type unit) {
    switch (unit) {
        case arrow::TimeUnit::NANO:   return value / 1'000'000LL;
        case arrow::TimeUnit::MICRO:  return value / 1'000LL;
        case arrow::TimeUnit::MILLI:  return value;
        case arrow::TimeUnit::SECOND: return value * 1'000LL;
    }
    return value;
}

// Detect the TimeUnit from a field that is either timestamp or int64.
arrow::TimeUnit::type detect_unit(const std::shared_ptr<arrow::Field>& field) {
    if (field->type()->id() == arrow::Type::TIMESTAMP) {
        auto ts_type = std::static_pointer_cast<arrow::TimestampType>(field->type());
        return ts_type->unit();
    }
    return arrow::TimeUnit::MILLI;
}

// Open a parquet file and return a FileReader.
std::unique_ptr<parquet::arrow::FileReader> open_file(const std::string& path) {
    auto file_result = arrow::io::ReadableFile::Open(path);
    if (!file_result.ok()) {
        throw std::runtime_error("Cannot open parquet file: " + path
                                 + " — " + file_result.status().ToString());
    }
    auto infile = file_result.ValueUnsafe();

    auto reader_result = parquet::arrow::OpenFile(infile, arrow::default_memory_pool());
    if (!reader_result.ok()) {
        throw std::runtime_error("Cannot read parquet: " + path
                                 + " — " + reader_result.status().ToString());
    }
    return std::move(reader_result).ValueUnsafe();
}

// Safe column index lookup.
int col_index(const std::shared_ptr<arrow::Schema>& schema,
              const std::string& name,
              const std::string& file_path) {
    int idx = schema->GetFieldIndex(name);
    if (idx < 0) {
        throw std::runtime_error("Column '" + name + "' not found in " + file_path);
    }
    return idx;
}

// Helper: iterate over chunks of a ChunkedArray, reading int64 timestamp values
// and converting to milliseconds. Writes results into `out` starting at `offset`.
void read_ts_column(const std::shared_ptr<arrow::ChunkedArray>& chunked,
                    arrow::TimeUnit::type unit,
                    std::vector<int64_t>& out) {
    out.reserve(static_cast<size_t>(chunked->length()));
    for (const auto& chunk : chunked->chunks()) {
        // Both TimestampArray and Int64Array store int64 internally.
        auto* arr = static_cast<const arrow::Int64Array*>(chunk.get());
        for (int64_t i = 0; i < arr->length(); ++i) {
            out.push_back(arr->IsNull(i) ? -1 : timestamp_to_ms(arr->Value(i), unit));
        }
    }
}

// Read a double column into a flat vector.
void read_double_column(const std::shared_ptr<arrow::ChunkedArray>& chunked,
                        std::vector<double>& out) {
    out.reserve(static_cast<size_t>(chunked->length()));
    for (const auto& chunk : chunked->chunks()) {
        auto* arr = static_cast<const arrow::DoubleArray*>(chunk.get());
        for (int64_t i = 0; i < arr->length(); ++i) {
            out.push_back(arr->IsNull(i) ? 0.0 : arr->Value(i));
        }
    }
}

// Read a boolean column into a flat vector.
void read_bool_column(const std::shared_ptr<arrow::ChunkedArray>& chunked,
                      std::vector<bool>& out) {
    out.reserve(static_cast<size_t>(chunked->length()));
    for (const auto& chunk : chunked->chunks()) {
        auto* arr = static_cast<const arrow::BooleanArray*>(chunk.get());
        for (int64_t i = 0; i < arr->length(); ++i) {
            out.push_back(!arr->IsNull(i) && arr->Value(i));
        }
    }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
void read_aggtrades(const std::string& path, const AggTickCallback& cb) {
    auto reader   = open_file(path);
    auto metadata = reader->parquet_reader()->metadata();
    int  n_rg     = metadata->num_row_groups();

    std::shared_ptr<arrow::Schema> schema;
    {
        auto status = reader->GetSchema(&schema);
        if (!status.ok()) {
            throw std::runtime_error("Cannot read schema: " + path);
        }
    }

    int ts_col    = col_index(schema, "timestamp", path);
    int price_col = col_index(schema, "price",     path);
    int qty_col   = col_index(schema, "quantity",  path);
    int buyer_col = schema->GetFieldIndex("is_buyer_maker"); // optional

    arrow::TimeUnit::type ts_unit = detect_unit(schema->field(ts_col));

    std::vector<int> col_indices = {ts_col, price_col, qty_col};
    if (buyer_col >= 0) col_indices.push_back(buyer_col);

    for (int rg = 0; rg < n_rg; ++rg) {
        std::shared_ptr<arrow::Table> table;
        auto status = reader->ReadRowGroup(rg, col_indices, &table);
        if (!status.ok()) {
            throw std::runtime_error("Error reading row group " + std::to_string(rg)
                                     + " of " + path + ": " + status.ToString());
        }

        // Read columns into flat vectors (handles multiple chunks).
        std::vector<int64_t> ts_vec;
        std::vector<double>  price_vec, qty_vec;
        std::vector<bool>    buyer_vec;

        read_ts_column(table->GetColumnByName("timestamp"), ts_unit, ts_vec);
        read_double_column(table->GetColumnByName("price"), price_vec);
        read_double_column(table->GetColumnByName("quantity"), qty_vec);

        if (buyer_col >= 0) {
            read_bool_column(table->GetColumnByName("is_buyer_maker"), buyer_vec);
        }

        size_t n_rows = ts_vec.size();
        std::vector<AggTick> batch;
        batch.reserve(n_rows);

        for (size_t i = 0; i < n_rows; ++i) {
            if (ts_vec[i] < 0) continue; // null timestamp

            AggTick tick;
            tick.timestamp_ms   = ts_vec[i];
            tick.price          = price_vec[i];
            tick.quantity       = qty_vec[i];
            tick.is_buyer_maker = buyer_col >= 0 && i < buyer_vec.size()
                                      ? buyer_vec[i]
                                      : false;
            batch.push_back(tick);
        }

        if (!batch.empty()) cb(batch);
    }
}

// ---------------------------------------------------------------------------
void read_bookticker(const std::string& path, const BookTickCallback& cb) {
    auto reader   = open_file(path);
    auto metadata = reader->parquet_reader()->metadata();
    int  n_rg     = metadata->num_row_groups();

    std::shared_ptr<arrow::Schema> schema;
    {
        auto status = reader->GetSchema(&schema);
        if (!status.ok()) {
            throw std::runtime_error("Cannot read schema: " + path);
        }
    }

    int ts_col  = col_index(schema, "transaction_time", path);
    int bbp_col = col_index(schema, "best_bid_price",   path);
    int bbq_col = col_index(schema, "best_bid_qty",     path);
    int bap_col = col_index(schema, "best_ask_price",   path);
    int baq_col = col_index(schema, "best_ask_qty",     path);

    arrow::TimeUnit::type ts_unit = detect_unit(schema->field(ts_col));

    std::vector<int> col_indices = {ts_col, bbp_col, bbq_col, bap_col, baq_col};

    for (int rg = 0; rg < n_rg; ++rg) {
        std::shared_ptr<arrow::Table> table;
        auto status = reader->ReadRowGroup(rg, col_indices, &table);
        if (!status.ok()) {
            throw std::runtime_error("Error reading row group " + std::to_string(rg)
                                     + " of " + path + ": " + status.ToString());
        }

        std::vector<int64_t> ts_vec;
        std::vector<double>  bbp_vec, bbq_vec, bap_vec, baq_vec;

        read_ts_column(table->GetColumnByName("transaction_time"), ts_unit, ts_vec);
        read_double_column(table->GetColumnByName("best_bid_price"), bbp_vec);
        read_double_column(table->GetColumnByName("best_bid_qty"),   bbq_vec);
        read_double_column(table->GetColumnByName("best_ask_price"), bap_vec);
        read_double_column(table->GetColumnByName("best_ask_qty"),   baq_vec);

        size_t n_rows = ts_vec.size();
        std::vector<BookTick> batch;
        batch.reserve(n_rows);

        for (size_t i = 0; i < n_rows; ++i) {
            if (ts_vec[i] < 0) continue;

            BookTick tick;
            tick.timestamp_ms   = ts_vec[i];
            tick.best_bid_price = bbp_vec[i];
            tick.best_bid_qty   = bbq_vec[i];
            tick.best_ask_price = bap_vec[i];
            tick.best_ask_qty   = baq_vec[i];
            batch.push_back(tick);
        }

        if (!batch.empty()) cb(batch);
    }
}

} // namespace io
