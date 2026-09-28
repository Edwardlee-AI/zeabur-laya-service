# Zeabur 部署設定（照 infinity-rag 同款）

## Option A：GHCR image（推薦，build 好晒啟動快）

- Source: **Image**
- Image: `ghcr.io/edwardlee-ai/zeabur-laya-service:latest`
- Service 名建議：`laya`（internal host 大概係 `laya.zeabur.internal`）
- Port: `8000`
- Health Check Path: `/health`

### 首次部署注意
Actions build 完之後，GHCR package 預設可能係 private。
兩條路揀一：
1. GitHub → 個人 profile → Packages → `zeabur-laya-service` → Package settings → Change visibility → **Public**（同 infinity-rag-deploy 一樣做法），Zeabur 唔使 credentials 直接拉。
2. 或者 Zeabur 設定 registry credentials（GitHub username + read:packages token）。

## Option B：GitHub repo 直接 build

- Source: GitHub
- Repository: `Edwardlee-AI/zeabur-laya-service`
- Branch: `main`
- Build Method: Dockerfile
- Port: `8000`
- Health Check Path: `/health`

## Env

```
LAYA_HOST=0.0.0.0
LAYA_PORT=8000
LAYA_THREADS=4
LAYA_API_KEY=<自訂一個密鑰>
```

可選：
- `LAYA_MODELS=multilingual`（只 preload 一個 checkpoint，慳 RAM，用喺 2 GiB plan）
- `LAYA_PRELOAD=0`（完全 lazy，第一個 request 先載入；唔建議，因為 /v1/systemone 會等好耐）

## Resource 建議

- 最低：2 vCPU / 2 GiB RAM（配 `LAYA_MODELS=multilingual`）
- 建議：2 vCPU / **4 GiB RAM**（全部三個 checkpoint preload，bf16 後實測 ~1.2–1.5 GiB 用量）

## 第一次啟動

- 會由 Hugging Face 落 ~1.5G checkpoint（`convaiinnovations/laya`，公開模型，唔使 token），需時幾分鐘。
- 載入期間 `/health` 答 `{"status":"loading"}`（HTTP 200），health check 唔會殺 pod。
- `/health` 變 `"status":"ok"` 同 `loaded` 有 checkpoint 名，先至真係 ready。

## 部署後驗證

```bash
curl -s http://<host>:8000/health

curl -s -X POST http://<host>:8000/v1/systemone \
  -H "Authorization: Bearer <LAYA_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"state":"我哋三月被扣咗兩次款，請今日退款，否則我哋會取消訂閱。","questions":{"urgency":{"type":"score","instructions":"How urgent is this?","criteria":["not urgent","soon","blocking"]}}}'
```

## OpenClaw 對接

部署完喺 Zeabur dashboard 抄個 internal hostname（例如 `laya.zeabur.internal`），
OpenClaw pod 行 HTTP 指過去（同 bge-m3 / infinity 一樣）：
`http://laya.zeabur.internal:8000/v1/systemone`。