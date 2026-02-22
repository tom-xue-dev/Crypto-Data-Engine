# 前端

React + Vite + TypeScript。包含 **Dollar Bar 蜡烛图可视化**。

## 开发

```bash
npm install
npm run dev
```

访问 http://localhost:5173 。确保后端已启动（`poetry run main start` 或 `uvicorn ...`），前端会通过 Vite 代理将 `/api` 请求转发到 `http://localhost:8000`。

## Dollar Bar 可视化

- 在页面输入 **交易对**（如 `BTCUSDT`）和 **根数**（如 500），点击「加载 K 线」。
- 后端从默认 bar 目录（`E:/data/dollar_bar/bars`）读取对应 symbol 的 parquet，返回 OHLCV，前端用 TradingView lightweight-charts 渲染蜡烛图。
- 若 bar 数据在其他目录，需在后端 API 中配置或通过接口参数 `bar_dir` 传入（当前前端未暴露该参数，可自行扩展）。

## 构建

```bash
npm run build
```

输出在 `dist/`，可配合任意静态服务器或后端挂载使用。
