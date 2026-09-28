#!/usr/bin/env python3
"""Entrypoint for the Zeabur LAYA service.

Stack:
1. bf16 load patches proven in tools/laya/test_v3.py (2026-09-27):
   default-dtype coercion, F.linear dtype alignment, chunked lazy
   safetensors load -> roughly halves checkpoint RAM and bounds load peak.
2. laya's official HTTP server (laya.serve): POST /v1/systemone (TypeSafe
   Jev wire protocol) + GET /health, optional bearer auth via LAYA_API_KEY.
3. A temporary /health responder during checkpoint download+load, so a
   container health check cannot kill the slow first boot (HF download of
   ~1.5G). uvicorn takes over the port once models are resident.
"""
import gc
import os
import threading
import time
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("LAYA_PORT", "8000"))
HOST = os.environ.get("LAYA_HOST", "0.0.0.0")


def _log(msg):
    print("[laya-entry] %s" % msg, flush=True)


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

    def do_GET(self):
        body = b'{"status":"loading"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


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


def main():
    tmp = _TmpServer((HOST, PORT), _LoadingHandler)
    threading.Thread(target=tmp.serve_forever, daemon=True).start()
    _log("temp /health responder up on :%d (status=loading)" % PORT)

    install_load_patches()
    from laya.serve import create_app
    t0 = time.time()
    app = create_app()  # preloads checkpoints (HF download on first boot)
    _log("create_app done in %.1fs" % (time.time() - t0))

    tmp.shutdown()
    tmp.server_close()
    _wait_port_free(PORT)
    _log("handing :%d to uvicorn" % PORT)

    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT,
                log_level=os.environ.get("LAYA_LOG_LEVEL", "info"))


if __name__ == "__main__":
    main()