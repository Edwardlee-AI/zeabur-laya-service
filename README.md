# zeabur-laya-service

[Laya](https://huggingface.co/convaiinnovations/laya) System-1 decision engine
as a long-running HTTP service for Zeabur — same deployment pattern as
`zeabur-infinity-rag-deploy` (bge-m3 embedding service): GitHub repo ->
Actions build -> GHCR image -> Zeabur service with its own RAM budget.

## What runs

- `laya` 0.3.20 official server (`laya.serve`): `POST /v1/systemone`
  (TypeSafe Jev wire protocol) + `GET /health`, optional `LAYA_API_KEY` bearer
  auth.
- `run_service.py` applies the bf16 + chunked-load stack proven in
  `tools/laya/test_v3.py` (halves checkpoint RAM), and answers `/health` with
  `{"status":"loading"}` during the first-boot model download so container
  health checks do not kill the pod before it is ready.

## Endpoints

- `GET /health` -> `{"status":"ok","loaded":[...],"device":"..."}`
- `POST /v1/systemone` -> `{"state": "...", "questions": {...}}` returns
  Jev-shaped `{model, answers, usage, routing}`.

Deploy settings: see `ZEABUR_DEPLOY.md`.