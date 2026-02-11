"""
Dollar bar factor test: step-by-step load, compute factors, verify I/O.

Step 1: Load dollar bar parquet(s), verify required columns.
Step 2: Run UnifiedFeatureCalculator (returns, volatility, momentum, volume, microstructure).
Step 3: Add VPIN, Kyle Lambda, MR, Trade Intensity, approximate OFI.
Step 4: Print factor summary table and optionally save sample CSV.
Step 5: Predictive power: Spearman correlation of each factor with forward 1/5/20-bar returns.

Usage:
    python scripts/run_dollarbar_factor_test.py
    python scripts/run_dollarbar_factor_test.py --bar-dir E:/data/dollar_bar/bars --symbol BTCUSDT
    python scripts/run_dollarbar_factor_test.py --max-files 2 --out-csv sample.csv
    python scripts/run_dollarbar_factor_test.py --all --max-files 12 --out-csv E:/data/backtest_results/dollarbar
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

REQUIRED_COLUMNS = [
    "start_time", "end_time", "open", "high", "low", "close",
    "volume", "buy_volume", "sell_volume", "vwap", "dollar_volume",
]
OPTIONAL_FOR_MICRO = ["path_efficiency", "impact_density", "tick_count"]

FACTOR_PREFIXES = (
    "return_", "momentum_", "volatility_", "path_efficiency_", "impact_density_",
    "signed_pe_", "VPIN_", "Kyle_Lambda_", "MR_", "lambda_", "OFI_approx_",
)


def get_factor_columns(df: pd.DataFrame) -> list:
    """Return sorted list of factor column names in df."""
    return sorted(
        c for c in df.columns
        if any(c.startswith(p) for p in FACTOR_PREFIXES)
        and df[c].dtype in (np.float64, np.float32)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dollar bar factor test: load, compute factors, verify each step"
    )
    parser.add_argument(
        "--bar-dir",
        default="E:/data/dollar_bar/bars",
        help="Root directory containing symbol subdirs with parquet files",
    )
    parser.add_argument(
        "--symbol",
        default=None,
        help="Symbol subdir name (e.g. BTCUSDT). If omitted, use first available.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=3,
        help="Max parquet files to load per symbol (default 3 for quick run)",
    )
    parser.add_argument(
        "--out-csv",
        default=None,
        help="If set, write sample rows (first + last 100) to this path. With --all, treat as directory and write {out-csv}/{symbol}.csv per symbol.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run pipeline for every symbol subdir in bar-dir (all trading pairs)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Step 1: Load dollar bar and verify input
# ---------------------------------------------------------------------------

def step1_load_and_verify(
    bar_dir: Path,
    symbol: Optional[str],
    max_files: int,
    quiet: bool = False,
) -> pd.DataFrame:
    """Load dollar bar parquets for one symbol; verify required columns."""
    if not quiet:
        print("\n" + "=" * 60)
        print("Step 1: Load dollar bar and verify input")
        print("=" * 60)

    if not symbol:
        subdirs = sorted(d for d in bar_dir.iterdir() if d.is_dir())
        if not subdirs:
            raise FileNotFoundError(f"No symbol subdirs in {bar_dir}")
        symbol = subdirs[0].name
        if not quiet:
            print(f"  [auto] Using first symbol: {symbol}")

    symbol_dir = bar_dir / symbol
    if not symbol_dir.exists():
        raise FileNotFoundError(f"Symbol dir not found: {symbol_dir}")

    files = sorted(symbol_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet in {symbol_dir}")

    to_load = files[:max_files]
    if not quiet:
        print(f"  Loading {len(to_load)} file(s) from {symbol_dir.name}/")

    frames = []
    for f in to_load:
        df = pd.read_parquet(f)
        if len(df) > 0:
            frames.append(df)
    if not frames:
        raise ValueError(f"No rows in any of {to_load}")

    df = pd.concat(frames, ignore_index=True)
    if "start_time" in df.columns:
        df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
        df = df.sort_values("start_time").reset_index(drop=True)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available: {list(df.columns)}"
        )

    if not quiet:
        print(f"  shape: {df.shape[0]} rows x {df.shape[1]} cols")
        print(f"  start_time range: {df['start_time'].min()} -> {df['start_time'].max()}")
        print(f"  columns: {list(df.columns)}")
        print("\n  head(3) [key cols]:")
        key_cols = [c for c in ["start_time", "close", "volume", "dollar_volume", "vwap"] if c in df.columns]
        print(df[key_cols].head(3).to_string())
        print()
    return df


# ---------------------------------------------------------------------------
# Step 2: UnifiedFeatureCalculator and verify output
# ---------------------------------------------------------------------------

def step2_unified_features(df: pd.DataFrame, quiet: bool = False) -> pd.DataFrame:
    """Compute existing features via UnifiedFeatureCalculator; verify output."""
    if not quiet:
        print("\n" + "=" * 60)
        print("Step 2: UnifiedFeatureCalculator (returns, vol, momentum, micro)")
        print("=" * 60)

    from crypto_data_engine.services.feature.unified_features import (
        UnifiedFeatureCalculator,
        UnifiedFeatureConfig,
    )

    config = UnifiedFeatureConfig(
        windows=[5, 10, 20, 60],
        include_returns=True,
        include_volatility=True,
        include_momentum=True,
        include_volume=True,
        include_microstructure=True,
        include_alphas=False,
        include_technical=False,
        include_cross_sectional=False,
    )
    calculator = UnifiedFeatureCalculator(config)
    out = calculator.calculate(df)

    expected = ["return_20", "momentum_20", "volatility_20"]
    for col in expected:
        if col not in out.columns:
            raise ValueError(f"Expected column missing after feature calc: {col}")
        non_null = out[col].notna().sum()
        if not quiet:
            if non_null == 0:
                print(f"  [WARN] {col} is all NaN")
            else:
                print(f"  {col}: non_null={non_null}, sample(last)={out[col].iloc[-1]}")

    if not quiet:
        optional_micro = ["path_efficiency_20", "impact_density_20"]
        for col in optional_micro:
            if col in out.columns:
                non_null = out[col].notna().sum()
                print(f"  {col}: non_null={non_null}, sample(last)={out[col].iloc[-1]}")
            else:
                print(f"  {col}: not present (bar may lack path_efficiency/impact_density)")
        new_cols = [c for c in out.columns if c not in df.columns]
        print(f"\n  New columns added: {len(new_cols)}")
        print("  " + ", ".join(new_cols[:20]) + (" ..." if len(new_cols) > 20 else ""))
        print()
    return out


# ---------------------------------------------------------------------------
# Step 3: Dollar-bar specific factors (VPIN, Kyle Lambda, MR, Trade Intensity, OFI)
# ---------------------------------------------------------------------------

def step3_dollar_bar_factors(df: pd.DataFrame, quiet: bool = False) -> pd.DataFrame:
    """Add VPIN, Kyle Lambda, MR, Trade Intensity, approximate OFI; verify each."""
    if not quiet:
        print("\n" + "=" * 60)
        print("Step 3: Dollar-bar factors (VPIN, Kyle Lambda, MR, lambda, OFI_approx)")
        print("=" * 60)

    out = df.copy()

    # VPIN(50): mean(|buy_volume - sell_volume| / volume) over 50 bars
    vpin_bar = (out["buy_volume"] - out["sell_volume"]).abs() / out["volume"].replace(0, np.nan)
    out["VPIN_50"] = vpin_bar.rolling(50).mean()
    if not quiet:
        n = out["VPIN_50"].notna().sum()
        print(f"  VPIN_50: non_null={n}, mean={out['VPIN_50'].mean():.6f}, last={out['VPIN_50'].iloc[-1]}")

    # Kyle Lambda(100): sum(|dclose|) / sum(dollar_volume)
    abs_ret = out["close"].diff().abs()
    out["Kyle_Lambda_100"] = (
        abs_ret.rolling(100).sum()
        / out["dollar_volume"].rolling(100).sum().replace(0, np.nan)
    )
    if not quiet:
        n = out["Kyle_Lambda_100"].notna().sum()
        print(f"  Kyle_Lambda_100: non_null={n}, mean={out['Kyle_Lambda_100'].mean():.2e}, last={out['Kyle_Lambda_100'].iloc[-1]}")

    # MR(20, 200): (VWAP_fast - VWAP_slow) / realized_vol_slow
    vwap_fast = out["vwap"].rolling(20).mean()
    vwap_slow = out["vwap"].rolling(200).mean()
    vol_slow = out["close"].pct_change().rolling(200).std()
    out["MR_20_200"] = (vwap_fast - vwap_slow) / vol_slow.replace(0, np.nan)
    if not quiet:
        n = out["MR_20_200"].notna().sum()
        print(f"  MR_20_200: non_null={n}, mean={out['MR_20_200'].mean():.4f}, last={out['MR_20_200'].iloc[-1]}")

    # Trade Intensity: tick_count / time_span (per bar then rolling), then z-score
    if "tick_count" in out.columns:
        time_span_sec = (out["end_time"] - out["start_time"]).dt.total_seconds().replace(0, np.nan)
        lambda_20 = out["tick_count"].rolling(20).sum() / time_span_sec.rolling(20).sum()
        out["lambda_20"] = lambda_20
        roll500_mean = lambda_20.rolling(500).mean()
        roll500_std = lambda_20.rolling(500).std()
        out["lambda_20_zscore"] = (lambda_20 - roll500_mean) / roll500_std.replace(0, np.nan)
        if not quiet:
            n = out["lambda_20"].notna().sum()
            print(f"  lambda_20: non_null={n}, mean={out['lambda_20'].mean():.2f}, last={out['lambda_20'].iloc[-1]}")
            nz = out["lambda_20_zscore"].notna().sum()
            print(f"  lambda_20_zscore: non_null={nz}, last={out['lambda_20_zscore'].iloc[-1]}")
    elif not quiet:
        print("  lambda_20 / lambda_20_zscore: skipped (no tick_count)")

    # Approximate OFI: (buy_volume - sell_volume) / (buy_volume + sell_volume), then rolling
    ofi_bar = (out["buy_volume"] - out["sell_volume"]) / (
        (out["buy_volume"] + out["sell_volume"]).replace(0, np.nan)
    )
    for w in [20, 100, 500]:
        col = f"OFI_approx_{w}"
        out[col] = ofi_bar.rolling(w).mean()
        if not quiet:
            n = out[col].notna().sum()
            print(f"  {col}: non_null={n}, mean={out[col].mean():.4f}, last={out[col].iloc[-1]}")

    if not quiet:
        print()
    return out


# ---------------------------------------------------------------------------
# Step 4: Factor summary and optional CSV export
# ---------------------------------------------------------------------------

def step4_summary_and_export(
    df: pd.DataFrame,
    out_csv: Optional[str],
    quiet: bool = False,
) -> None:
    """Print factor summary table; optionally write sample CSV."""
    if not quiet:
        print("\n" + "=" * 60)
        print("Step 4: Factor summary and optional export")
        print("=" * 60)

    factor_cols = get_factor_columns(df)

    if not factor_cols:
        if not quiet:
            print("  No factor columns found.")
        return

    if not quiet:
        rows = []
        for col in factor_cols:
            s = df[col]
            rows.append({
                "factor": col,
                "non_null": int(s.notna().sum()),
                "mean": round(s.mean(), 6) if s.notna().any() else None,
                "std": round(s.std(), 6) if s.notna().sum() > 1 else None,
            })
        summary = pd.DataFrame(rows)
        print(summary.to_string(index=False))
        print()

    if out_csv:
        path = Path(out_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        head = df.head(100)
        tail = df.tail(100)
        sample = pd.concat([head, tail], ignore_index=True)
        sample.to_csv(path, index=False)
        if not quiet:
            print(f"  Wrote sample ({len(sample)} rows) to {path}")
    if not quiet:
        print()


# ---------------------------------------------------------------------------
# Step 5: Predictive power (factor vs forward returns)
# ---------------------------------------------------------------------------

FORWARD_HORIZONS = (5, 10, 20, 50)


def step5_predictive_power(
    df: pd.DataFrame,
    quiet: bool = False,
) -> pd.DataFrame:
    """
    Compute Spearman correlation of each factor with forward bar returns.
    Returns a DataFrame with columns: factor, fwd5_corr, fwd10_corr, fwd20_corr, fwd50_corr, n_valid.
    """
    if "close" not in df.columns:
        if not quiet:
            print("\n  Step 5 skipped: no 'close' column.")
        return pd.DataFrame()

    for horizon in FORWARD_HORIZONS:
        col = f"_fwd_ret_{horizon}"
        df[col] = df["close"].shift(-horizon) / df["close"] - 1.0

    factor_cols = get_factor_columns(df)
    if not factor_cols:
        if not quiet:
            print("\n  Step 5: No factor columns.")
        for horizon in FORWARD_HORIZONS:
            df.drop(columns=[f"_fwd_ret_{horizon}"], inplace=True)
        return pd.DataFrame()

    rows = []
    for col in factor_cols:
        row = {"factor": col}
        n_valid = None
        for horizon in FORWARD_HORIZONS:
            fwd_col = f"_fwd_ret_{horizon}"
            valid = df[[col, fwd_col]].dropna()
            n = len(valid)
            if n_valid is None or n < n_valid:
                n_valid = n
            if n < 10:
                row[f"fwd{horizon}_corr"] = np.nan
            else:
                corr = valid[col].corr(valid[fwd_col], method="spearman")
                row[f"fwd{horizon}_corr"] = round(float(corr), 4) if not np.isnan(corr) else np.nan
        row["n_valid"] = n_valid
        rows.append(row)

    for horizon in FORWARD_HORIZONS:
        df.drop(columns=[f"_fwd_ret_{horizon}"], inplace=True)

    result = pd.DataFrame(rows)

    if not quiet:
        print("\n" + "=" * 60)
        print("Step 5: Predictive power (Spearman corr vs forward returns)")
        print("=" * 60)
        print("  fwdN_corr = correlation of factor at t with return from t to t+N bars.")
        print(result.to_string(index=False))
        print()

    return result


def main() -> None:
    args = parse_args()
    bar_dir = Path(args.bar_dir)

    if args.all:
        subdirs = sorted(d for d in bar_dir.iterdir() if d.is_dir())
        symbols = [d.name for d in subdirs if list(d.glob("*.parquet"))]
        if not symbols:
            raise FileNotFoundError(
                f"No symbol subdirs with parquet files in {bar_dir}"
            )
        out_csv_dir = Path(args.out_csv) if args.out_csv else None
        if out_csv_dir is not None:
            out_csv_dir.mkdir(parents=True, exist_ok=True)
        summary_rows = []
        per_symbol_pred_dfs = []
        for idx, symbol in enumerate(symbols):
            out_path = None
            if out_csv_dir is not None:
                out_path = str(out_csv_dir / f"{symbol}.csv")
            try:
                df = step1_load_and_verify(
                    bar_dir=bar_dir,
                    symbol=symbol,
                    max_files=args.max_files,
                    quiet=True,
                )
                n_rows = len(df)
                df = step2_unified_features(df, quiet=True)
                df = step3_dollar_bar_factors(df, quiet=True)
                n_factors = len(get_factor_columns(df))
                step4_summary_and_export(df, out_path, quiet=True)
                pred_df = step5_predictive_power(df, quiet=True)
                if not pred_df.empty:
                    pred_df = pred_df.copy()
                    pred_df["symbol"] = symbol
                    per_symbol_pred_dfs.append(pred_df)
                avg_abs_corr = np.nan
                corr_cols = [c for c in pred_df.columns if c.startswith("fwd") and c.endswith("_corr")] if not pred_df.empty else []
                if corr_cols:
                    abs_corrs = pred_df[corr_cols].abs()
                    avg_abs_corr = round(float(abs_corrs.max(axis=1).mean()), 4)
                summary_rows.append({
                    "symbol": symbol,
                    "rows": n_rows,
                    "factors": n_factors,
                    "avg_|corr|": avg_abs_corr if not np.isnan(avg_abs_corr) else "",
                    "status": "OK",
                    "out_csv": out_path or "",
                })
            except Exception as exc:
                summary_rows.append({
                    "symbol": symbol,
                    "rows": "",
                    "factors": "",
                    "avg_|corr|": "",
                    "status": str(exc)[:80],
                    "out_csv": "",
                })
        summary_df = pd.DataFrame(summary_rows)
        print("\n" + "=" * 60)
        print("Summary (--all)")
        print("=" * 60)
        print(summary_df.to_string(index=False))
        ok = sum(1 for r in summary_rows if r["status"] == "OK")
        print(f"\n  {ok}/{len(symbols)} symbols OK.")

        if per_symbol_pred_dfs:
            all_pred = pd.concat(per_symbol_pred_dfs, ignore_index=True)
            corr_cols = [c for c in all_pred.columns if c.startswith("fwd") and c.endswith("_corr")]
            if corr_cols:
                cross_rows = []
                for _, row in all_pred.groupby("factor"):
                    factor_name = row["factor"].iloc[0]
                    out_row = {"factor": factor_name}
                    for col in corr_cols:
                        vals = row[col].dropna()
                        out_row[f"{col}_mean"] = round(vals.mean(), 4) if len(vals) > 0 else np.nan
                        out_row[f"{col}_std"] = round(vals.std(), 4) if len(vals) > 1 else np.nan
                        out_row[f"{col}_n"] = int(vals.count())
                    cross_rows.append(out_row)
                cross_df = pd.DataFrame(cross_rows)
                print("\n" + "=" * 60)
                print("Factor summary across assets (Spearman corr vs forward returns)")
                print("=" * 60)
                print("  mean/std/n = cross-symbol mean, std, and count of valid symbols per factor.")
                print(cross_df.to_string(index=False))
                print()
        return

    df = step1_load_and_verify(
        bar_dir=bar_dir,
        symbol=args.symbol,
        max_files=args.max_files,
    )
    df = step2_unified_features(df)
    df = step3_dollar_bar_factors(df)
    step4_summary_and_export(df, args.out_csv)
    step5_predictive_power(df)
    print("Done. All steps completed.")


if __name__ == "__main__":
    main()
