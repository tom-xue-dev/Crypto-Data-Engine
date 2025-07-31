# Tick 数据下载接口文档

本接口文档描述了基于 FastAPI + Celery 构建的 Tick 数据下载系统的用户接口。

---

## ✅ 接口总览

| 接口路径 | 方法 | 作用 |
|----------|------|------|
| `/download-tick` | `POST` | 提交 tick 下载任务，返回 `task_id` |
| `/task-status/{task_id}` | `GET` | 查询任务状态与结果 |
| `/download-history` | `GET` | 查询当前用户历史下载任务记录（分页） |

---

## 📥 1. POST `/download-tick`

### 请求参数（JSON）:

```json
{
  "symbol": "BTCUSDT",
  "start_time": "2023-01-01T00:00:00",
  "end_time": "2023-01-02T00:00:00"
}
```

### 响应示例:

```json
{
  "task_id": "2ec1e57a-3f48-4c96-91b0-4fcfe90d1df5",
  "status": "PENDING"
}
```

---

## 📥 2. GET `/task-status/{task_id}`

用于查询任务执行状态及下载结果路径。

### 响应示例（任务未完成）:

```json
{
  "task_id": "2ec1e57a-3f48-4c96-91b0-4fcfe90d1df5",
  "status": "PENDING"
}
```

### 响应示例（任务完成）:

```json
{
  "task_id": "2ec1e57a-3f48-4c96-91b0-4fcfe90d1df5",
  "status": "SUCCESS",
  "result": {
    "file_path": "/data/btcusdt/20230101.csv"
  }
}
```

---

## 📥 3. GET `/download-history`

分页查询当前用户历史下载记录。

### 请求参数（Query）:

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `page` | int | 页码，默认 1 |
| `page_size` | int | 每页数量，默认 20 |
| `symbol` | str | 可选，筛选指定交易对 |

### 响应示例:

```json
{
  "total": 132,
  "records": [
    {
      "task_id": "....",
      "symbol": "BTCUSDT",
      "created_at": "2023-01-01T00:00:00",
      "status": "SUCCESS",
      "file_path": "/data/btcusdt/20230101.csv",
      "duration": 10.3
    }
  ]
}
```

---
