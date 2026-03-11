"""
Factor report generation. No computation, no data loading.

Input:  analysis metrics dict (from FactorAnalyzer)
Output: PNG charts, CSV files, summary table
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crypto_data_engine.common.logger.logger import get_logger

logger = get_logger(__name__)


class FactorReporter:
    """Generate charts and export analysis results to disk."""

    def summary_table(
        self, batch_results: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """Create a comparison table across all analyzed factors.

        Sorted by absolute ICIR of the first period.
        """
        rows = []
        for name, result in batch_results.items():
            row: Dict[str, Any] = {"factor": name}

            mean_ic = result.get("mean_ic")
            ic_ir = result.get("ic_ir")
            if mean_ic is not None:
                for period, val in mean_ic.items():
                    row[f"IC_{period}"] = round(val, 4)
            if ic_ir is not None:
                for period, val in ic_ir.items():
                    row[f"ICIR_{period}"] = round(val, 4)

            qr = result.get("quantile_returns")
            if qr is not None and not qr.empty:
                q_min, q_max = qr.index.min(), qr.index.max()
                for col in qr.columns:
                    spread = qr.loc[q_max, col] - qr.loc[q_min, col]
                    row[f"spread_{col}"] = round(spread, 6)

            bt = result.get("backtest")
            if bt is not None:
                perf = bt["performance"]
                row["bt_return"] = round(perf.get("total_return", 0), 4)
                row["bt_annual"] = round(perf.get("annual_return", 0), 4)
                row["bt_sharpe"] = round(perf.get("sharpe_ratio", 0), 2)
                row["bt_mdd"] = round(perf.get("max_drawdown", 0), 4)
                row["bt_calmar"] = round(perf.get("calmar_ratio", 0), 2)

            rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("factor")
        icir_cols = [c for c in df.columns if c.startswith("ICIR_")]
        if icir_cols:
            df = df.sort_values(icir_cols[0], key=abs, ascending=False)
        return df

    # ------------------------------------------------------------------
    # PNG tear sheets
    # ------------------------------------------------------------------

    def create_tear_sheet(
        self,
        factor_name: str,
        metrics: Dict[str, Any],
        output_dir: Path,
    ) -> List[Path]:
        """Generate IC / quantile / turnover PNGs for one factor."""
        factor_dir = output_dir / factor_name
        factor_dir.mkdir(parents=True, exist_ok=True)
        saved: List[Path] = []

        saved += self._chart_ic(factor_name, metrics, factor_dir)
        saved += self._chart_cumulative_ic(factor_name, metrics, factor_dir)
        saved += self._chart_quantile_returns(factor_name, metrics, factor_dir)
        saved += self._chart_turnover(factor_name, metrics, factor_dir)
        saved += self._chart_backtest_nav(factor_name, metrics, factor_dir)

        logger.info(f"Saved {len(saved)} charts for '{factor_name}' -> {factor_dir}")
        return saved

    def create_batch_tear_sheets(
        self,
        batch_results: Dict[str, Dict[str, Any]],
        output_dir: Path,
    ) -> Dict[str, List[Path]]:
        """Generate PNGs for all factors."""
        all_paths: Dict[str, List[Path]] = {}
        total = len(batch_results)
        for i, (name, result) in enumerate(batch_results.items(), 1):
            logger.info(f"[{i}/{total}] Creating charts for: {name}")
            all_paths[name] = self.create_tear_sheet(name, result, output_dir)
        return all_paths

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    def export(
        self,
        batch_results: Dict[str, Dict[str, Any]],
        output_dir: Path,
        charts: bool = True,
    ):
        """Export summary + per-factor CSV + optional PNGs."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Summary
        summary = self.summary_table(batch_results)
        summary.to_csv(output_dir / "summary.csv")
        logger.info(f"Summary table saved -> {output_dir / 'summary.csv'}")

        # Per-factor CSV
        for name, result in batch_results.items():
            factor_dir = output_dir / name
            factor_dir.mkdir(parents=True, exist_ok=True)

            ic = result.get("ic")
            if ic is not None:
                ic.to_csv(factor_dir / "ic.csv")

            qr = result.get("quantile_returns")
            if qr is not None:
                qr.to_csv(factor_dir / "quantile_returns.csv")

            bt = result.get("backtest")
            if bt is not None:
                bt["nav"].to_csv(factor_dir / "backtest_nav.csv", header=True)
                bt["returns"].to_csv(factor_dir / "backtest_returns.csv")
                # Save performance summary
                perf_s = pd.Series(bt["performance"], name="value")
                perf_s.to_csv(factor_dir / "backtest_performance.csv")

        # PNG charts
        if charts:
            self.create_batch_tear_sheets(batch_results, output_dir)

        logger.info(f"All results exported -> {output_dir}")

    # ------------------------------------------------------------------
    # Chart helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _chart_ic(
        factor_name: str, metrics: Dict[str, Any], out_dir: Path
    ) -> List[Path]:
        ic = metrics.get("ic")
        if ic is None or ic.empty:
            return []
        try:
            fig, axes = plt.subplots(
                len(ic.columns), 1,
                figsize=(14, 4 * len(ic.columns)),
                squeeze=False,
            )
            for i, col in enumerate(ic.columns):
                ax = axes[i, 0]
                ic_s = ic[col].dropna()
                ax.bar(ic_s.index, ic_s.values,
                       alpha=0.5, width=(ic_s.index[1] - ic_s.index[0]) * 0.8 if len(ic_s) > 1 else 1.0,
                       color="steelblue")
                rolling = ic_s.rolling(min(20, len(ic_s))).mean()
                ax.plot(rolling.index, rolling.values,
                        color="red", linewidth=2, label="Rolling IC (20)")
                mean_ic = ic_s.mean()
                ax.axhline(y=mean_ic, color="black", linestyle="--",
                           label=f"Mean IC: {mean_ic:.4f}")
                ax.set_title(f"{factor_name} — IC ({col}-bar forward)")
                ax.legend(loc="upper right")
                ax.set_ylabel("IC (Spearman)")

            plt.tight_layout()
            path = out_dir / "ic.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return [path]
        except Exception as exc:
            logger.warning(f"IC chart failed for {factor_name}: {exc}")
            plt.close("all")
            return []

    @staticmethod
    def _chart_cumulative_ic(
        factor_name: str, metrics: Dict[str, Any], out_dir: Path
    ) -> List[Path]:
        cum_ic = metrics.get("cumulative_ic")
        if cum_ic is None or cum_ic.empty:
            return []
        try:
            fig, ax = plt.subplots(figsize=(14, 5))
            for col in cum_ic.columns:
                s = cum_ic[col].dropna()
                ax.plot(s.index, s.values, label=f"{col}-period", linewidth=1.5)
            ax.axhline(y=0, color="black", linewidth=0.5)
            ax.set_title(f"{factor_name} — Cumulative IC")
            ax.set_ylabel("Cumulative IC")
            ax.set_xlabel("Date")
            ax.legend(loc="upper left")
            plt.tight_layout()
            path = out_dir / "cumulative_ic.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return [path]
        except Exception as exc:
            logger.warning(f"Cumulative IC chart failed for {factor_name}: {exc}")
            plt.close("all")
            return []

    @staticmethod
    def _chart_quantile_returns(
        factor_name: str, metrics: Dict[str, Any], out_dir: Path
    ) -> List[Path]:
        qr = metrics.get("quantile_returns")
        if qr is None or qr.empty:
            return []
        try:
            fig, axes = plt.subplots(
                1, len(qr.columns),
                figsize=(5 * len(qr.columns), 5),
                squeeze=False,
            )
            for i, col in enumerate(qr.columns):
                ax = axes[0, i]
                values = qr[col]
                colors = ["green" if v > 0 else "red" for v in values]
                x = range(len(values))
                ax.bar(x, values.values, color=colors)
                ax.set_xticks(x, labels=[str(v) for v in values.index])
                ax.set_title(f"{col}-bar forward return")
                ax.set_xlabel("Quantile")
                ax.set_ylabel("Mean Return")
                ax.axhline(y=0, color="black", linewidth=0.5)

            fig.suptitle(f"{factor_name} — Quantile Returns", fontsize=14)
            plt.tight_layout()
            path = out_dir / "quantile_returns.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return [path]
        except Exception as exc:
            logger.warning(f"Quantile chart failed for {factor_name}: {exc}")
            plt.close("all")
            return []

    @staticmethod
    def _chart_turnover(
        factor_name: str, metrics: Dict[str, Any], out_dir: Path
    ) -> List[Path]:
        turnover = metrics.get("turnover", {})
        if not turnover:
            return []
        try:
            fig, ax = plt.subplots(figsize=(14, 4))
            for period, t_series in turnover.items():
                if isinstance(t_series, pd.DataFrame):
                    mean_t = t_series.mean(axis=1)
                else:
                    mean_t = t_series
                ax.plot(range(len(mean_t)), mean_t.values, label=f"{period}-bar")
            ax.set_title(f"{factor_name} — Quantile Turnover")
            ax.set_ylabel("Turnover")
            ax.legend()
            plt.tight_layout()
            path = out_dir / "turnover.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return [path]
        except Exception as exc:
            logger.warning(f"Turnover chart failed for {factor_name}: {exc}")
            plt.close("all")
            return []

    @staticmethod
    def _chart_backtest_nav(
        factor_name: str, metrics: Dict[str, Any], out_dir: Path
    ) -> List[Path]:
        bt = metrics.get("backtest")
        if bt is None:
            return []
        nav = bt.get("nav")
        if nav is None or nav.empty:
            return []
        try:
            perf = bt["performance"]
            cfg = bt["config"]
            fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

            # --- NAV curve ---
            ax = axes[0]
            ax.plot(nav.index, nav.values, color="steelblue", linewidth=1.5)
            ax.axhline(y=1.0, color="black", linewidth=0.5, linestyle="--")
            ax.fill_between(
                nav.index, 1.0, nav.values,
                where=nav.values >= 1.0, alpha=0.15, color="green",
            )
            ax.fill_between(
                nav.index, 1.0, nav.values,
                where=nav.values < 1.0, alpha=0.15, color="red",
            )
            label = (
                f"Return: {perf.get('total_return', 0):.2%}  "
                f"Annual: {perf.get('annual_return', 0):.2%}  "
                f"Sharpe: {perf.get('sharpe_ratio', 0):.2f}  "
                f"MDD: {perf.get('max_drawdown', 0):.2%}  "
                f"Cost: {cfg.cost_bps}bps"
            )
            ax.set_title(
                f"{factor_name} — Long Q{perf.get('long_quantile', '?')} / "
                f"Short Q{perf.get('short_quantile', '?')}",
                fontsize=13,
            )
            ax.set_ylabel("NAV")
            ax.legend([label], loc="upper left", fontsize=9)

            # --- Drawdown ---
            ax2 = axes[1]
            cummax = nav.cummax()
            dd = (nav - cummax) / cummax
            ax2.fill_between(dd.index, 0, dd.values, color="red", alpha=0.4)
            ax2.set_ylabel("Drawdown")
            ax2.set_xlabel("Date")

            plt.tight_layout()
            path = out_dir / "backtest_nav.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return [path]
        except Exception as exc:
            logger.warning(f"Backtest NAV chart failed for {factor_name}: {exc}")
            plt.close("all")
            return []
