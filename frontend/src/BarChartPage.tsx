import { useCallback, useState } from "react";
import BarChart from "./BarChart";

const API_BASE = "/api";

type BarsResponse = {
  symbol: string;
  bar_dir: string;
  bars: Array<{
    time: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume?: number;
    dollar_volume?: number;
  }>;
  count: number;
};

export default function BarChartPage() {
  const [symbol, setSymbol] = useState("BTCUSDT");
  const [limit, setLimit] = useState(500);
  const [bars, setBars] = useState<BarsResponse["bars"] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadBars = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({ symbol, limit: String(limit) });
      const res = await fetch(`${API_BASE}/viz/bars?${params}`);
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail ?? res.statusText);
      }
      const data: BarsResponse = await res.json();
      setBars(data.bars);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setBars(null);
    } finally {
      setLoading(false);
    }
  }, [symbol, limit]);

  return (
    <section className="bar-chart-page">
      <div className="controls">
        <label>
          交易对
          <input
            type="text"
            value={symbol}
            onChange={(e) => setSymbol(e.target.value)}
            placeholder="BTCUSDT"
          />
        </label>
        <label>
          根数
          <input
            type="number"
            min={1}
            max={5000}
            value={limit}
            onChange={(e) => setLimit(Number(e.target.value) || 500)}
          />
        </label>
        <button type="button" onClick={loadBars} disabled={loading}>
          {loading ? "加载中…" : "加载 K 线"}
        </button>
      </div>
      {error && <p className="error">{error}</p>}
      {bars && bars.length > 0 && (
        <div className="chart-container">
          <BarChart data={bars} symbol={symbol} />
        </div>
      )}
      {bars && bars.length === 0 && !loading && (
        <p className="hint">暂无数据，请检查交易对或 bar 目录。</p>
      )}
    </section>
  );
}
