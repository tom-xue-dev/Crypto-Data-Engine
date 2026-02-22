import { useEffect, useRef } from "react";
import {
  createChart,
  CandlestickSeries,
  ColorType,
} from "lightweight-charts";
import type { IChartApi, ISeriesApi, CandlestickData } from "lightweight-charts";

type BarItem = {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
};

function parseTime(iso: string): number {
  const date = new Date(iso);
  return Math.floor(date.getTime() / 1000);
}

function toCandlestickData(bars: BarItem[]): CandlestickData[] {
  return bars.map((b) => ({
    time: parseTime(b.time),
    open: b.open,
    high: b.high,
    low: b.low,
    close: b.close,
  })) as CandlestickData[];
}

type BarChartProps = {
  data: BarItem[];
  symbol: string;
};

export default function BarChart({ data, symbol }: BarChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);

  useEffect(() => {
    if (!containerRef.current || data.length === 0) return;

    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "#1a1a1a" },
        textColor: "#d1d4dc",
      },
      grid: {
        vertLines: { color: "#2b2b2b" },
        horzLines: { color: "#2b2b2b" },
      },
      width: containerRef.current.clientWidth,
      height: 500,
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
        borderColor: "#2b2b2b",
      },
      rightPriceScale: {
        borderColor: "#2b2b2b",
        scaleMargins: {
          top: 0.1,
          bottom: 0.2,
        },
      },
    });

    const candlestickSeries = chart.addSeries(CandlestickSeries, {
      upColor: "#26a69a",
      downColor: "#ef5350",
      borderVisible: false,
      wickUpColor: "#26a69a",
      wickDownColor: "#ef5350",
    });

    const chartData = toCandlestickData(data);
    candlestickSeries.setData(chartData);

    chartRef.current = chart;
    seriesRef.current = candlestickSeries;

    const handleResize = () => {
      if (containerRef.current && chartRef.current) {
        chartRef.current.applyOptions({ width: containerRef.current.clientWidth });
      }
    };
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
    };
  }, [data]);

  return (
    <div className="bar-chart-wrapper">
      <h2>{symbol} Dollar Bar</h2>
      <div ref={containerRef} className="bar-chart" />
    </div>
  );
}
