#!/usr/bin/env python3
"""Entrypoint for the Zeabur LAYA service.

Stack:
1. bf16 load patches proven in tools/laya/test_v3.py (2026-09-27):
   default-dtype coercion, F.linear dtype alignment, chunked lazy
   safetensors load -> roughly halves checkpoint RAM and bounds load peak.
2. Memory-aware checkpoint selection (2026-09-28 crash-loop fix):
   reads the cgroup memory limit and preloads only the checkpoints that
   fit. The 13:38 UTC first deploy proved all-three CAN load (health ok),
   but the pod died seconds later and crash-looped -- classic peak-RAM OOM
   at the exact moment preload completes. Do not trust the operator to
   set LAYA_MODELS; decide here, from the real limit.
3. laya's official HTTP server (laya.serve): POST /v1/systemone (TypeSafe
   Jev wire protocol) + GET /health, optional bearer auth via LAYA_API_KEY.
4. A temporary /health responder during checkpoint download+load, so a
   container health check cannot kill the slow first boot (HF download of
   ~1.5G). GET answers 200 {"status":"loading"}; POST answers 503 so
   clients can tell "still loading" from "real error". uvicorn takes over
   the port once models are resident.
"""
import gc
import os
import threading
import time
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("LAYA_PORT", "8000"))
HOST = os.environ.get("LAYA_HOST", "0.0.0.0")

GIB = 1 << 30


def _log(msg):
    print("[laya-entry] %s" % msg, flush=True)


# ---------------------------------------------------------------------------
# 0. memory helpers (cgroup v2 -> v1 -> /proc/meminfo)
# ---------------------------------------------------------------------------
def detect_memory_limit_bytes():
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(path) as f:
                raw = f.read().strip()
            if raw in ("max", ""):
                continue
            val = int(raw)
            if val > (1 << 50):  # v1 "unlimited" sentinel
                continue
            return val
        except (OSError, ValueError):
            continue
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def _mem_rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return -1


def choose_checkpoints(limit_bytes):
    """Pick the checkpoint set that fits the pod with sane headroom.

    All three checkpoints are ~1.16B params (~2.2 GiB bf16 weights) plus
    ~0.6-1 GiB Python/torch/transformers runtime. Observed 2026-09-28:
    all-three peaks right at a small plan's ceiling and OOM-kills the
    container the moment preload finishes.
    """
    if limit_bytes <= 0:
        return ["english", "multilingual"]
    gib = limit_bytes / GIB
    if gib < 1.5:
        _log("WARNING: %.2f GiB limit too small for multilingual; english only" % gib)
        return ["english"]
    if gib < 3.0:
        return ["multilingual"]
    if gib < 5.5:
        return ["english", "multilingual"]
    return ["english", "multilingual", "typed-decisions"]


# ---------------------------------------------------------------------------
# 1. bf16 + lazy chunked load patches (from test_v3, pod-proven)
# ---------------------------------------------------------------------------
def install_load_patches():
    import torch

    raw = os.environ.get("LAYA_THREADS", "")
    if raw.strip().isdigit() and int(raw) > 0:
        torch.set_num_threads(int(raw))
    torch.set_grad_enabled(False)

    _orig_sdd = torch.set_default_dtype

    def _coerce_sdd(dtype):
        if dtype == torch.float32:
            _log("set_default_dtype(fp32) coerced to bf16")
            dtype = torch.bfloat16
        _orig_sdd(dtype)

    torch.set_default_dtype = _coerce_sdd

    _orig_linear = torch.nn.functional.linear

    def _linear_cast(x, w, b=None):
        if x.dtype != w.dtype:
            x = x.to(w.dtype)
        return _orig_linear(x, w, b)

    torch.nn.functional.linear = _linear_cast
    torch.set_default_dtype(torch.bfloat16)
    _log("torch %s ready (bf16 default)" % torch.__version__)

    from safetensors import safe_open
    import safetensors.torch as _st

    class TensorProxy:
        def __init__(self, f, name):
            self._f = f
            self._name = name
            try:
                self._sl = f.get_slice(name)
                self.shape = tuple(int(d) for d in self._sl.get_shape())
                self.dtype = self._sl.get_dtype()
            except Exception:
                self._sl = None
                t = f.get_tensor(name)
                self.shape = tuple(int(d) for d in t.shape)
                self.dtype = t.dtype

        def copy_into_(self, p):
            if self._sl is None or not self.shape:
                p.data.copy_(self._f.get_tensor(self._name))
                return
            if isinstance(self.dtype, torch.dtype):
                es = torch.empty((), dtype=self.dtype).element_size()
            else:
                es = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8,
                      "I32": 4, "I16": 2, "I8": 1, "U8": 1,
                      "BOOL": 1}.get(str(self.dtype).upper(), 2)
            row = es
            for d in self.shape[1:]:
                row *= d
            step = max(1, (2 << 20) // max(row, 1))
            n = self.shape[0]
            dst = p.data
            for i in range(0, n, step):
                dst[i:i + step].copy_(self._sl[i:i + step])

    class LazySD(Mapping):
        def __init__(self, path):
            self._f = safe_open(path, framework="pt", device="cpu")
            self._keys = list(self._f.keys())
            self._keyset = set(self._keys)

        def __getitem__(self, k):
            if k not in self._keyset:
                raise KeyError(k)
            return TensorProxy(self._f, k)

        def __iter__(self):
            return iter(self._keys)

        def __len__(self):
            return len(self._keys)

        def __contains__(self, k):
            return k in self._keyset

        def keys(self):
            return list(self._keys)

    _st.load_file = lambda path, device="cpu", **kw: LazySD(path)

    _orig_lsd = torch.nn.Module.load_state_dict

    def _patched_lsd(self, state_dict, strict=True, assign=False):
        if not isinstance(state_dict, LazySD):
            return _orig_lsd(self, state_dict, strict=strict, assign=assign)
        from torch.nn.modules.module import _IncompatibleKeys
        own = dict(self.named_parameters())
        own.update(dict(self.named_buffers()))
        expected = list(self.state_dict().keys())
        expected_set = set(expected)
        missing = [n for n in expected if n not in state_dict]
        unexpected = [k for k in state_dict.keys() if k not in expected_set]
        _log("load_state_dict begin (%d tensors)" % len(expected))
        done = 0
        for name in expected:
            if name in state_dict:
                state_dict[name].copy_into_(own[name])
                done += 1
                if done % 100 == 0:
                    _log("load_state_dict %d/%d" % (done, len(expected)))
        gc.collect()
        _log("load_state_dict end")
        if strict and (missing or unexpected):
            raise RuntimeError(
                "strict load failure: missing=%r unexpected=%r"
                % (missing[:8], unexpected[:8]))
        return _IncompatibleKeys(missing, unexpected)

    torch.nn.Module.load_state_dict = _patched_lsd
    _log("lazy chunked load patch active")


# ---------------------------------------------------------------------------
# 2. temp /health responder while checkpoints download + load
# ---------------------------------------------------------------------------
class _LoadingHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _reply(self, code):
        body = b'{"status":"loading"}'
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._reply(200)

    def do_POST(self):
        # 503 so smoke tests distinguish "still loading" from a real error,
        # instead of http.server's opaque 501 "Unsupported method".
        self._reply(503)


class _TmpServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def _wait_port_free(port, attempts=30):
    import socket
    for _ in range(attempts):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", port))
            s.close()
            return True
        except OSError:
            s.close()
            time.sleep(0.3)
    return False


def _pin_router(router, pin):
    """Restrict routing (and therefore loading) to the models in `pin`.

    laya's Router lazy-loads whichever checkpoint a request routes to, and
    `preload()` raises `max_loaded` to fit the preloaded set, so an English
    request against a multilingual-only deployment builds the english
    checkpoint on demand. Two resident checkpoints peaked ~2.6 GiB here and
    the shared 8 GiB Zeabur pool (OpenClaw ~3 GiB + ClickHouse + other
    services + this one) OOM-killed the container within a minute (observed
    2026-09-28 19:09). Pinning keeps exactly the preloaded set resident;
    other traffic still works, just on the pinned checkpoint (English on
    multilingual scores ~0.66 MASSIVE intent vs ~0.78 on the english
    checkpoint -- a fair trade for a pod that stays up).
    """
    from laya.router import _repo_str
    allowed = {m for m in pin if m in getattr(router, "models", {})}
    unknown = [m for m in pin if m not in allowed]
    if unknown:
        _log("LAYA_PIN ignoring unknown models: %s" % unknown)
    if not allowed:
        _log("LAYA_PIN: nothing valid to pin; router left unpinned")
        return
    repos = {k: _repo_str(router.models[k]) for k in allowed}
    primary = sorted(allowed)[0]
    orig_route = router._route

    def _pinned_route(state, questions=None, model=None, task=None, lang=None, lang_guess=None):
        d = orig_route(state, questions, model=model, task=task, lang=lang, lang_guess=lang_guess)
        if d.get("model") not in allowed:
            reason = d.get("reason")
            d = dict(d)
            d["model"] = primary
            d["repo"] = repos[primary]
            d["reason"] = "pinned to %r by LAYA_PIN; original route: %s" % (primary, reason)
        return d

    router._route = _pinned_route
    router.max_loaded = len(allowed)
    _log("LAYA_PIN active: only %s can ever be resident" % sorted(allowed))


def main():
    tmp = _TmpServer((HOST, PORT), _LoadingHandler)
    threading.Thread(target=tmp.serve_forever, daemon=True).start()
    _log("temp /health responder up on :%d (status=loading)" % PORT)

    install_load_patches()
    _log("rss after patches: %d MiB" % _mem_rss_mb())

    # Memory-aware checkpoint selection BEFORE laya.serve reads the env.
    # Explicit LAYA_MODELS always wins.
    if not os.environ.get("LAYA_MODELS", "").strip():
        limit = detect_memory_limit_bytes()
        chosen = choose_checkpoints(limit)
        os.environ["LAYA_MODELS"] = ",".join(chosen)
        _log("auto checkpoint selection: limit=%s -> %s"
             % (("%.2f GiB" % (limit / GIB)) if limit else "unknown", chosen))
    else:
        chosen = [m.strip() for m in os.environ["LAYA_MODELS"].split(",") if m.strip()]
        _log("LAYA_MODELS set by operator: %r" % os.environ["LAYA_MODELS"])

    # Pin routing to the preload set (LAYA_PIN overrides; none/off/* disables).
    # Without this, a request in another script lazy-loads a second checkpoint:
    # two resident peaked ~2.6 GiB and the shared pool OOM-killed the pod
    # within 60s (observed 2026-09-28 19:09).
    pin_env = os.environ.get("LAYA_PIN", "").strip()
    if pin_env.lower() in ("none", "off", "*"):
        pin = []
    elif pin_env:
        pin = [m.strip() for m in pin_env.split(",") if m.strip()]
    else:
        pin = chosen

    from laya.serve import build_router, create_app
    t0 = time.time()
    router = build_router()  # preloads the chosen checkpoints (HF download on first boot)
    if pin:
        _pin_router(router, pin)
    app = create_app(router=router)
    _log("create_app done in %.1fs, rss=%d MiB" % (time.time() - t0, _mem_rss_mb()))

    tmp.shutdown()
    tmp.server_close()
    _wait_port_free(PORT)
    _log("handing :%d to uvicorn, rss=%d MiB" % (PORT, _mem_rss_mb()))

    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT,
                log_level=os.environ.get("LAYA_LOG_LEVEL", "info"))


if __name__ == "__main__":
    main()