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
- `LAYA_MODELS` 而家**唔使設**：自動讀 container memory limit 揀（見下）。手動設咗就以你嘅為準。
- `LAYA_PIN`：限制 router 永遠淨用 preload 嘅 checkpoint（預設開）。設 `none`／`off` 關閉。
  呢個係 2026-09-28 第二次 OOM 嘅修復：英文 request 會 lazy-load 第二個 checkpoint，
  兩個齊載 ~2.6G 喺共享 8G pool 度 60 秒內被殺（實測 19:09）。釘死後英文照答，
  只係行 multilingual（英文 MASSIVE intent 0.657 vs 0.783，換 pod 唔死）。
- `LAYA_PRELOAD=0`（完全 lazy，第一個 request 先載入；唔建議，因為 /v1/systemone 會等好耐）

### 自動 checkpoint 選擇（2026-09-28 fix）

⚠️ 8GB 共享 pool 實測教訓（19:09）：`loaded:["multilingual"]` 穩行一個鐘，一有英文 request
lazy-load 埋 english → 兩個齊載 ~2.6G → 60 秒內 OOM 被殺。所以而家 default 釘死 LAYA_PIN。

`LAYA_MODELS` 未設時，entrypoint 讀 cgroup memory limit 自動揀：

| Plan RAM | 載入 | 峰值約 |
|---|---|---|
| ≥5.5 GiB | english + multilingual + typed-decisions | ~3 GiB |
| 3–5.5 GiB | english + multilingual | ~2.2 GiB |
| 1.5–3 GiB | multilingual（支援中英） | ~1.4 GiB |
| <1.5 GiB | english only（警告） | ~1 GiB |

背景：首次部署三個全載曾經成功報 `ok` 但隨即 crash loop — 峰值貼頂 OOM。自動揀就唔會咁。

### 載入期間嘅回應

- `GET /health` → 200 `{"status":"loading"}`（health check 唔會殺 pod）
- `POST /v1/systemone` → 503 `{"status":"loading"}`（唔再係 http.server 嗰個 501 HTML）

## Volume（強烈建議）

掛一個 persistent volume 喺 **`/app/.cache`**（HF_HOME）。
冇 volume：每次 restart 都要重新由 Hugging Face 下載 ~1.5G checkpoint（要幾分鐘，crash loop 時更慘）。
有 volume：落一次之後 restart 秒載。

## Resource 建議

- 最低：2 vCPU / 2 GiB RAM（自動淨載 multilingual）
- 建議：2 vCPU / **4 GiB RAM**（自動載 english + multilingual，中英都齊）
- 想三個 checkpoint 全載（連 typed-decisions）：8 GiB

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