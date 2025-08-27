# API 文档 (README)

统一前缀：`/api/v1`  
返回格式：JSON  
认证：JWT（可选），幂等操作支持 `Idempotency-Key` 头  

---

## 下载服务 (Downloader, IO bound)

### 源管理
- `GET /api/v1/sources`  
- `POST /api/v1/sources/{source}/credentials`  
- `GET /api/v1/sources/{source}/symbols`

### 下载任务
- `POST /api/v1/downloads/jobs`  
- `GET /api/v1/downloads/jobs/{job_id}`  
- `GET /api/v1/downloads/tasks`  
- `GET /api/v1/downloads/tasks/{task_id}`  
- `POST /api/v1/downloads/tasks/{task_id}:cancel`  
- `POST /api/v1/downloads/tasks/{task_id}:retry`  
- `POST /api/v1/downloads/bulk:cancel`  
- `POST /api/v1/downloads/bulk:retry`  
- `GET /api/v1/downloads/metrics`

### 数据产物
- `GET /api/v1/catalog/datasets`  
- `POST /api/v1/catalog/datasets`  
- `GET /api/v1/files`  
- `POST /api/v1/files/verify`  
- `POST /api/v1/files/compact`

### 实时与回调
- `GET /api/v1/stream/downloads` (WebSocket/SSE)  
- `POST /api/v1/hooks/downloads`

---

## 聚合与特征服务 (Aggregator & Features, CPU bound)

### 聚合
- `POST /api/v1/aggregation/jobs`  
- `GET /api/v1/aggregation/jobs/{job_id}`  
- `POST /api/v1/aggregation/jobs/{job_id}:cancel`  
- `GET /api/v1/aggregation/metrics`

### 特征工程
- `GET /api/v1/features/registry`  
- `POST /api/v1/features/jobs`  
- `GET /api/v1/features/jobs/{job_id}`  
- `POST /api/v1/features/jobs/{job_id}:cancel`

### 依赖与血缘
- `GET /api/v1/lineage/{artifact_name}`  
- `POST /api/v1/cache:invalidate`

### 实时与回调
- `GET /api/v1/stream/compute` (WebSocket/SSE)  
- `POST /api/v1/hooks/compute`

---

## 回测服务 (Backtester)

### 策略管理
- `GET /api/v1/strategies`  
- `POST /api/v1/strategies`  
- `GET /api/v1/strategies/{strategy_id}`

### 回测任务
- `POST /api/v1/backtests`  
- `GET /api/v1/backtests/{bt_id}`  
- `GET /api/v1/backtests/{bt_id}/results`  
- `GET /api/v1/backtests/{bt_id}/charts/equity`  
- `POST /api/v1/backtests/{bt_id}:cancel`  
- `POST /api/v1/backtests/grid`  
- `GET /api/v1/backtests/grid/{grid_id}`

### 复现与快照
- `GET /api/v1/runs/{run_id}`  
- `POST /api/v1/runs/{run_id}:replay`

### 实时与回调
- `GET /api/v1/stream/backtests` (WebSocket/SSE)  
- `POST /api/v1/hooks/backtests`

---

## 信号与因子服务 (Signals & Factors)

- `POST /api/v1/signals/jobs`  
- `GET /api/v1/signals/jobs/{job_id}`  
- `GET /api/v1/signals/samples`

---

## 运维与集群 (Ops)

- `GET /api/v1/healthz`  
- `GET /api/v1/readyz`  
- `GET /api/v1/info`  
- `GET /api/v1/cluster`  
- `POST /api/v1/cluster/scale`  
- `GET /api/v1/queue/metrics`  
- `GET /api/v1/logs`  
- `GET /api/v1/config`  
- `POST /api/v1/config:reload`

---

## 认证与租户 (Auth & Tenant)

- `POST /api/v1/auth/login`  
- 多租户支持：所有资源支持 `tenant_id` 字段
