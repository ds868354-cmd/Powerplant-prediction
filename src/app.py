"""
app.py -- AUTO-GENERATED production serving script for TABULAR_ML.
This file contains ONLY the code needed to serve this one model type.

Layer A: replays the EXACT preprocessing sequence recorded in feature_recipe.json's
         "applied_pipeline_steps" ledger (impute -> winsorize -> scale -> one-hot) with NumPy + stdlib only.
Layer B: ONNX Runtime, a pure-NumPy linear-model JSON reader, or a pure-NumPy tree-ensemble JSON reader
         (LightGBM/XGBoost export) -- whichever artifact is present. NO pickle/joblib.
Also: sliding window of the last 100 inference rows + live PSI vs drift_baseline.json; when PSI crosses 0.2
      the response carries an "X-Model-Drift-Warning: true" header.
Endpoint: POST /predict/tabular

Prediction endpoints are ASYNC and run the CPU-bound scoring step via asyncio.to_thread so the event
loop stays responsive under concurrent request load.
"""
import asyncio
import collections
import glob
import hashlib
import hmac
import json
import os
import re
import shutil
import tempfile
import threading
import time

import numpy as np
from fastapi import FastAPI, BackgroundTasks, HTTPException, Response, Header, Depends, Request
from pydantic import BaseModel

# [DOCS FIX] /docs, /redoc and /openapi.json are served WITHOUT an API key (they are not behind require_api_key), so in
# production they handed every visitor the full request schema + feature names. They are now off whenever APP_ENV is
# production/prod; set ENABLE_DOCS=true to switch them back on (or run with APP_ENV=staging/dev locally).
_DOCS_ON = (os.environ.get("ENABLE_DOCS", "").strip().lower() in ("1", "true", "yes", "on")
            or os.environ.get("APP_ENV", "").strip().lower() not in ("production", "prod"))
_DOCS_KW = {} if _DOCS_ON else {"docs_url": None, "redoc_url": None, "openapi_url": None}


app = FastAPI(title="Model Serving API (TABULAR_ML)", **_DOCS_KW)

MODEL_TYPE = "TABULAR_ML"
# [DEPLOYMENT FIX] Every *_PATH below used to be the FULL path recorded at training time (e.g.
# "/content/output/rnn_model.onnx" from a Colab run) baked in verbatim -- correct on the training
# machine, but Docker/k8s copy artifacts next to THIS file (e.g. /app/), not into the original
# output_dir, so os.path.isfile(...) was always False at serving time and every artifact silently
# failed to load. generate_app_py() now passes only each artifact's basename (see bundler._bundle_relpath),
# and every path below is resolved against this script's own directory instead -- the identical bundle
# then works unmodified from Colab, a bare checkout, or a container, with no environment-specific path
# ever baked in. BUNDLE_DIR_OVERRIDE (env var) is an escape hatch for the rare case artifacts are
# deliberately staged somewhere else.
_BUNDLE_DIR = os.environ.get("BUNDLE_DIR_OVERRIDE") or os.path.dirname(os.path.abspath(__file__))


def _artifact_roots():
    """Directories searched for artifacts: the bundle directory, then its artifacts/ sub-folder. The Dockerfile copies the whole
    (allow-listed) build context, so a flat bundle and one that keeps its weights under artifacts/ both work."""
    return (_BUNDLE_DIR, os.path.join(_BUNDLE_DIR, "artifacts"))


def _resolve_bundle_path(name, default=None):
    """[PATH-RESOLUTION FIX] Resolve an artifact path against this script's own directory (then its artifacts/ sub-folder).

    * `name` given   -> <bundle dir>/<name>, or <bundle dir>/artifacts/<name> when only that one exists.
    * `name` empty   -> the caller did not know the artifact at generation time (e.g. the bundle was rebuilt
      from a folder, or the file was added later). Instead of returning "" (artifact silently never loaded)
      or the bare bundle directory, look for `default` -- an exact filename, a glob, or a tuple of either -- in the
      bundle dir and in artifacts/.
    * Nothing matches -> "" (callers already treat "" as "artifact not shipped").
    The result is therefore either "" or an existing file/directory, never the bundle directory itself."""
    if name:
        for _root in _artifact_roots():
            _p = os.path.join(_root, name)
            if os.path.exists(_p):
                return _p
        return os.path.join(_BUNDLE_DIR, name)
    for pattern in ((default,) if isinstance(default, str) else (default or ())):
        for _root in _artifact_roots():
            candidate = os.path.join(_root, pattern)
            is_glob = any(c in pattern for c in "*?[")
            hits = [h for h in (sorted(glob.glob(candidate)) if is_glob else ([candidate] if os.path.exists(candidate) else []))
                    if os.path.abspath(h) != os.path.abspath(_root)]
            if is_glob and len(hits) > 1:
                # never guess between several candidate files -- serving the wrong model is worse than not starting
                print("[Startup] WARNING: %d files match %s in the bundle (%s) -- ambiguous, none loaded. Ship exactly one, "
                      "or regenerate app.py so it names the file." % (len(hits), pattern,
                                                                      ", ".join(os.path.basename(h) for h in hits)))
                continue
            if hits:
                return hits[0]
    return ""

API_KEYS_PATH = _resolve_bundle_path("api_keys.json", "api_keys.json")
# [SECRETS FIX] api_keys.json is NOT baked into the Docker image / committed to git: a deployer mounts it (k8s Secret volume,
# `docker run -v`) and points API_KEYS_PATH at the mounted file. The copy next to app.py stays the default for local runs.
_API_KEYS_PATH_ENV = os.environ.get("API_KEYS_PATH", "").strip()
if _API_KEYS_PATH_ENV:
    API_KEYS_PATH = _API_KEYS_PATH_ENV


def _env_int(name, default):
    """[ENV-PARSE FIX] a typo such as RATE_LIMIT_PER_MINUTE=sixty used to raise ValueError at import time -> the pod crashed
    in a restart loop before it could even answer /health. A bad value now falls back to the default, with a startup warning."""
    raw = os.environ.get(name, "")
    if not str(raw).strip():
        return default
    try:
        return int(float(str(raw).strip()))
    except ValueError:
        print("[Startup] WARNING: " + name + "=" + repr(raw)[:40] + " is not a number -- using the default " + str(default) + ".")
        return default


# [LOAD-ROBUSTNESS FIX] a truncated / hand-edited / wrong-version JSON artifact used to raise at IMPORT time (KeyError, JSONDecodeError)
# -> uvicorn never bound a port and the pod sat in CrashLoopBackOff with the cause buried in the last log lines. Artifacts are now
# loaded through this helper: a failure is logged, remembered in _ARTIFACT_LOAD_ERRORS, shown on /health (503 + the reason) and
# the app still starts, so `kubectl describe` / the readiness probe point straight at the broken file.
_ARTIFACT_LOAD_ERRORS = {}


def _load_json_artifact(path, label, build=None):
    if not (path and os.path.isfile(path)):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return build(data) if build else data
    except Exception as e:  # noqa: BLE001 -- any parse / schema problem is reported, never raised at import time
        _ARTIFACT_LOAD_ERRORS[label] = type(e).__name__ + ": " + str(e)[:200]
        print("[Startup] WARNING: could not load " + str(path) + " -- " + _ARTIFACT_LOAD_ERRORS[label])
        return None


# [DOS FIX v2] nothing bounded the request body: one multi-GB POST (or a huge base64 image) was read fully into memory. The old guard only
# looked at the Content-Length HEADER, so a chunked / streamed upload (no header, or a lying one) bypassed it. This pure-ASGI middleware
# refuses a declared length above MAX_BODY_BYTES up front AND counts the bytes of a streamed body as they arrive (HTTP 413 either way).
# The default is sized per model type at generation time (small for JSON-only models, larger for image uploads); override with the env var.
_MAX_BODY_DEFAULT = 1000000
MAX_BODY_BYTES = _env_int("MAX_BODY_BYTES", _MAX_BODY_DEFAULT)


async def _send_413(send):
    _body = json.dumps(dict(detail="Request body too large (limit " + str(MAX_BODY_BYTES) + " bytes).")).encode("utf-8")
    await send(dict(type="http.response.start", status=413,
                    headers=[(b"content-type", b"application/json"), (b"content-length", str(len(_body)).encode("ascii"))]))
    await send(dict(type="http.response.body", body=_body))


class _BodyLimitMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        _cl = ""
        for _k, _v in scope.get("headers") or []:
            if _k == b"content-length":
                _cl = _v.decode("latin-1").strip()
                break
        if _cl.isdigit() and int(_cl) > MAX_BODY_BYTES:
            await _send_413(send)
            return
        _seen = [0]

        async def _limited_receive():
            _msg = await receive()
            if _msg["type"] == "http.request":
                _seen[0] += len(_msg.get("body", b""))
                if _seen[0] > MAX_BODY_BYTES:
                    # an HTTPException raised while the route reads the body is turned into a clean 413 JSON response
                    raise HTTPException(413, "Request body too large (limit " + str(MAX_BODY_BYTES) + " bytes).")
            return _msg

        await self.app(scope, _limited_receive, send)


app.add_middleware(_BodyLimitMiddleware)

# [HEALTH REDACTION] /health is unauthenticated (probes need it), so the raw exception text of a failed artifact load (file paths,
# parser messages) must not be handed to every visitor: only the artifact LABELS are shown. The full reason is always in the pod log;
# HEALTH_VERBOSE=true restores the detailed text for a staging debug session.
_HEALTH_VERBOSE = os.environ.get("HEALTH_VERBOSE", "").strip().lower() in ("1", "true", "yes", "on")


def _public_err(msg):
    """Text that may be shown to a caller. Raw exception messages (library internals, file paths) stay in the pod log unless
    HEALTH_VERBOSE=true; used by every unauthenticated / client-facing error that used to echo them."""
    if msg is None:
        return None
    return str(msg)[:300] if _HEALTH_VERBOSE else "details are in the server log (set HEALTH_VERBOSE=true to show them here)"


# ======================================================================================================================
# [RULE 4] NON-BLOCKING MONITORING -- latency, sliding windows and PSI drift run AFTER the response, never inside it.
# check_rate_limit() (every prediction endpoint depends on it) stamps the start time and queues _observe_request() as a FastAPI
# BackgroundTask: Starlette runs it once the response has been sent. The request path therefore pays only for one perf_counter()
# call and one list append -- no window copy, no histogram, no PSI. The windows are fixed-size deques (DRIFT_WINDOW_SIZE, default 200).
# ======================================================================================================================
def _env_float(name, default):
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return float(default)


_DRIFT_WINDOW_SIZE = max(10, _env_int("DRIFT_WINDOW_SIZE", 200))
_DRIFT_MIN_WINDOW = max(10, min(_DRIFT_WINDOW_SIZE, _env_int("DRIFT_MIN_WINDOW", 100)))    # PSI on fewer rows is noise
_DRIFT_EVAL_EVERY = max(1, _env_int("DRIFT_EVAL_EVERY", max(1, _DRIFT_WINDOW_SIZE // 4)))  # PSI recomputed at most once per N rows
_DRIFT_PSI_THRESHOLD = _env_float("DRIFT_PSI_THRESHOLD", 0.2)
_LOG_LATENCY = os.environ.get("LOG_LATENCY", "").strip().lower() in ("1", "true", "yes", "on")

_monitor_lock = threading.Lock()
_latency_window = collections.deque(maxlen=_DRIFT_WINDOW_SIZE)     # seconds, newest last
_drift_rows = collections.deque(maxlen=_DRIFT_WINDOW_SIZE)         # preprocessed feature rows (tabular-style models only)
_monitor_counters = {"observed": 0, "since_eval": 0}
_drift_state = {"drifted": False, "max_psi": None, "worst_feature_index": None, "window_rows": 0, "evaluated_at": None}
_drift_eval_lock = threading.Lock()
_drift_evaluator = [None]    # set by the tabular drift block: callable(2-D array) -> (max_psi, worst_feature_index)


def _evaluate_drift(snapshot):
    """Runs the PSI maths on a snapshot of the window. Background only; one evaluation at a time (a busy evaluator is skipped)."""
    _ev = _drift_evaluator[0]
    if _ev is None or not _drift_eval_lock.acquire(blocking=False):
        return
    try:
        _arr = np.asarray(snapshot, dtype=np.float64)
        _max_psi, _worst = _ev(_arr)
        with _monitor_lock:
            _drift_state.update(drifted=bool(_max_psi > _DRIFT_PSI_THRESHOLD), max_psi=round(float(_max_psi), 4),
                                worst_feature_index=int(_worst), window_rows=int(_arr.shape[0]), evaluated_at=time.time())
    except Exception as _e:  # noqa: BLE001 -- monitoring must never affect serving
        print("[Monitor] drift evaluation failed (ignored): " + type(_e).__name__ + ": " + str(_e)[:200])
    finally:
        _drift_eval_lock.release()


def _record_observation(started, row=None):
    """BACKGROUND work for one served request: latency into the sliding window, the feature row into the drift window, and -- every
    DRIFT_EVAL_EVERY rows once DRIFT_MIN_WINDOW rows exist -- a PSI evaluation."""
    try:
        _latency = max(0.0, time.perf_counter() - started) if started is not None else None
        if row is not None:
            row = np.asarray(row, dtype=np.float64).ravel().tolist()
        _snapshot = None
        with _monitor_lock:
            if _latency is not None:
                _latency_window.append(_latency)
            _monitor_counters["observed"] += 1
            if row is not None and _drift_evaluator[0] is not None:
                _drift_rows.append(row)
                _monitor_counters["since_eval"] += 1
                if len(_drift_rows) >= _DRIFT_MIN_WINDOW and _monitor_counters["since_eval"] >= _DRIFT_EVAL_EVERY:
                    _monitor_counters["since_eval"] = 0
                    _snapshot = list(_drift_rows)
        if _LOG_LATENCY and _latency is not None:
            print("[Latency] %.2f ms" % (_latency * 1000.0))
        if _snapshot is not None:
            _evaluate_drift(_snapshot)
    except Exception as _e:  # noqa: BLE001
        print("[Monitor] observation failed (ignored): " + type(_e).__name__ + ": " + str(_e)[:200])


def _observe_request(request):
    """Background task body: pick up what the request path left on request.state and record it."""
    _st = getattr(request, "state", None)
    _record_observation(getattr(_st, "monitor_t0", None), getattr(_st, "monitor_row", None))


def _monitoring_snapshot():
    with _monitor_lock:
        _lat = sorted(_latency_window)
        _observed = _monitor_counters["observed"]
        _drift = dict(_drift_state)
    def _pct(p):
        return round(_lat[min(len(_lat) - 1, int(p * len(_lat)))] * 1000.0, 3) if _lat else None
    _names = None
    _rec = globals().get("_recipe")
    if isinstance(_rec, dict) and _drift.get("worst_feature_index") is not None:
        _fn = _rec.get("feature_names") or []
        if 0 <= _drift["worst_feature_index"] < len(_fn):
            _names = _fn[_drift["worst_feature_index"]]
    _drift["worst_feature"] = _names
    _drift.update(monitoring_active=_drift_evaluator[0] is not None, psi_threshold=_DRIFT_PSI_THRESHOLD, min_window=_DRIFT_MIN_WINDOW)
    return {"window_size": _DRIFT_WINDOW_SIZE, "observations_total": _observed,
            "latency_ms": {"count": len(_lat), "p50": _pct(0.50), "p95": _pct(0.95), "max": round(_lat[-1] * 1000.0, 3) if _lat else None},
            "drift": _drift}


# ======================================================================================================================
# [RULE 3] SAFE ONNX + LARGE-WEIGHT LOADING -- companion-file (.onnx.data / sharded weights) resolution that never crashes the import.
# ======================================================================================================================
_ORT_PROVIDERS_ENV = [p.strip() for p in os.environ.get("ORT_PROVIDERS", "CPUExecutionProvider").split(",") if p.strip()]


def _onnx_external_names(model_path):
    """File names the ONNX graph asks for as external data (e.g. 'model.onnx.data'). They are plain strings inside the protobuf, so a
    bounded byte scan finds them without the `onnx` package. Only used on the rare repair path."""
    try:
        if os.path.getsize(model_path) > 256 * 1024 * 1024:
            return []
        with open(model_path, "rb") as _f:
            _blob = _f.read()
        _names = {m.decode("ascii", "ignore") for m in re.findall(rb"[A-Za-z0-9_.\-]{1,200}\.(?:onnx\.data|onnx_data|data)(?![A-Za-z0-9_])", _blob)}
        return sorted(n for n in _names if "/" not in n)
    except OSError:
        return []


def _stage_onnx(model_path, data_src, needed_name, mode):
    """Writable staging directory (TMPDIR) holding the model and its external data under the name the graph expects -- for a read-only
    root filesystem (the Kubernetes default here) where nothing can be repaired next to the model. mode 'link' symlinks both files
    (no extra disk); mode 'copy' copies them (when the runtime refuses symlinked external data)."""
    _stage = tempfile.mkdtemp(prefix="onnx_stage_")
    _sm, _sd = os.path.join(_stage, os.path.basename(model_path)), os.path.join(_stage, needed_name)
    try:
        if mode == "link":
            os.symlink(os.path.abspath(model_path), _sm)
            os.symlink(os.path.abspath(data_src), _sd)
        else:
            _need = os.path.getsize(model_path) + os.path.getsize(data_src)
            if shutil.disk_usage(_stage).free < _need * 1.1:
                raise OSError("not enough free space in the staging directory")
            shutil.copyfile(model_path, _sm)
            shutil.copyfile(data_src, _sd)
    except OSError:
        shutil.rmtree(_stage, ignore_errors=True)
        raise
    return _sm


def _load_onnx_session(ort_module, model_path):
    """Returns (session, None) or (None, reason) and NEVER raises, so a missing / renamed / unreadable ONNX file or companion can no
    longer crash the import (CrashLoopBackOff before uvicorn binds a port). Steps: (1) load as shipped; (2) when that fails because the
    external-data companion has a different name than the graph asks for ('rnn_model_onnx.data' vs 'rnn_model.onnx.data'), and exactly one
    other *.data / *.onnx_data file sits next to the model, give it the expected name -- in place (hard link, then copy) when the
    directory is writable, otherwise through a staging directory in TMPDIR (symlinks, then copies). Zero or several candidates are never
    guessed between."""
    if ort_module is None:
        return None, "onnxruntime is not installed in this image"
    if not (model_path and os.path.isfile(model_path)):
        return None, "ONNX model file not found: " + os.path.basename(str(model_path))
    try:
        _avail = ort_module.get_available_providers()
    except Exception:  # noqa: BLE001
        _avail = ["CPUExecutionProvider"]
    _providers = [p for p in _ORT_PROVIDERS_ENV if p in _avail] or ["CPUExecutionProvider"]

    def _try(path):
        try:
            return ort_module.InferenceSession(path, providers=_providers), None
        except Exception as _e:  # noqa: BLE001
            return None, type(_e).__name__ + ": " + str(_e)[:300]

    _sess, _err = _try(model_path)
    if _sess is not None:
        return _sess, None
    _errors = [_err]
    _dir = os.path.dirname(model_path)
    _wanted = _onnx_external_names(model_path) or [os.path.basename(model_path) + ".data"]
    if len(_wanted) != 1:
        return None, _errors[0] + " | external-data repair skipped: the graph references %d data files" % len(_wanted)
    _needed = _wanted[0]
    _needed_path = os.path.join(_dir, _needed)
    if os.path.isfile(_needed_path):
        return None, _errors[0]            # the companion is there under the right name: the failure is something else
    _cands = sorted(set(glob.glob(os.path.join(_dir, "*.data")) + glob.glob(os.path.join(_dir, "*.onnx_data"))))
    if len(_cands) != 1:
        return None, _errors[0] + " | external-data repair skipped: %d candidate file(s) next to the model" % len(_cands)
    _src = _cands[0]
    for _how in ("hardlink", "copy"):                       # 1) repair in place (writable filesystem)
        try:
            if _how == "hardlink":
                os.link(_src, _needed_path)
            else:
                shutil.copyfile(_src, _needed_path)
        except OSError:
            continue
        _sess, _err = _try(model_path)
        if _sess is not None:
            print("[Startup] Self-heal: ONNX external-data file '%s' provided as '%s' (%s)" % (os.path.basename(_src), _needed, _how))
            return _sess, None
        _errors.append(_err)
        try:
            os.remove(_needed_path)
        except OSError:
            pass
    for _mode in ("link", "copy"):                           # 2) read-only filesystem -> stage in TMPDIR
        try:
            _staged = _stage_onnx(model_path, _src, _needed, _mode)
        except OSError as _e:
            _errors.append("staging (%s) failed: %s" % (_mode, str(_e)[:120]))
            continue
        _sess, _err = _try(_staged)
        if _sess is not None:
            print("[Startup] Self-heal: ONNX model staged in %s with '%s' as '%s' (%s)" % (os.path.dirname(_staged), os.path.basename(_src), _needed, _mode))
            return _sess, None
        _errors.append(_err)
        shutil.rmtree(os.path.dirname(_staged), ignore_errors=True)
    return None, " | ".join(e for e in _errors if e)[:700]


def _check_weight_dir(path):
    """Problems (list of strings; empty = fine) with a local transformers / peft / sentence-transformers weights directory: no weight
    file at all, or a sharded checkpoint whose index lists shard files that are missing. No weight file is opened here."""
    if not os.path.isdir(path):
        return ["directory not found: " + os.path.basename(str(path))]
    _names = os.listdir(path)
    _probs = []
    if not [n for n in _names if n.endswith((".safetensors", ".bin", ".pt", ".pth"))]:
        _probs.append("no weight file (*.safetensors / *.bin) in " + os.path.basename(str(path)))
    for _n in _names:
        if _n.endswith(".index.json"):
            try:
                with open(os.path.join(path, _n), "r", encoding="utf-8") as _f:
                    _wm = json.load(_f).get("weight_map", {})
            except (OSError, ValueError, AttributeError):
                _probs.append("unreadable shard index " + _n)
                continue
            _missing = sorted({v for v in _wm.values() if isinstance(v, str) and not os.path.isfile(os.path.join(path, v))})
            if _missing:
                _probs.append("shard file(s) listed in %s are missing: %s" % (_n, ", ".join(_missing[:5])))
    return _probs


def _hf_safe_kwargs(path):
    """use_safetensors=True whenever the local directory ships *.safetensors, so a pickle-based *.bin next to it is never chosen."""
    try:
        if os.path.isdir(path) and any(n.endswith(".safetensors") for n in os.listdir(path)):
            return {"use_safetensors": True}
    except OSError:
        pass
    return {}



# ---- API-key authentication -------------------------------------------
# SECURITY FIX: every prediction endpoint used to have NO
# authentication at all -- auth.py already implemented APIKeyStore
# (salted HMAC-SHA256 key hashing, never plaintext) and even shipped a
# FASTAPI_AUTH_DEPENDENCY_SNIPPET meant to be wired into this file, but
# generate_app_py() never actually called it, so every generated app.py
# served predictions to anyone who could reach the port. This closes
# that gap without pulling in auth.py's file-locking machinery (that
# module's write-path concurrency guarantees matter for issuing/revoking
# keys, not for this read-only verification at request time): this file
# reads api_keys.json directly and verifies the same salted-hash format
# APIKeyStore.create_key() writes.
#
# Provisioning a key (run once, from the bundle folder -- create_api_key.py ships with the bundle):
#   python create_api_key.py --owner my-service
#   # -> save the printed key now; it is never recoverable from api_keys.json again
#
# Callers then send it as:  -H "X-API-Key: <raw_key>"
#
# REQUIRE_AUTH is tri-state (true / false / auto) -- see the block below.
# [AUTH FALLBACK FIX] REQUIRE_AUTH is now tri-state instead of "true unless exactly 'false'":
#   REQUIRE_AUTH=true  (or 1/yes/on)  -> always enforce. Missing api_keys.json => clear 503 (server misconfigured),
#                                        not a confusing 401 for every caller.
#   REQUIRE_AUTH=false (or 0/no/off)  -> never enforce (local / staging).
#   unset or "auto"                   -> enforce when api_keys.json exists and holds at least one key; if it does
#                                        not exist, fall back to open access with a loud startup warning -- EXCEPT
#                                        when APP_ENV=production, which stays fail-closed.
# /health always reports the effective value as "auth_enforced" so the mode is never a surprise.
_REQUIRE_AUTH_RAW = os.environ.get("REQUIRE_AUTH", "auto").strip().lower() or "auto"
_APP_ENV = os.environ.get("APP_ENV", "").strip().lower()


# [RULE 2] SECRET SOURCES + GRACEFUL FAILURE. The key store is read from a MOUNTED FILE (k8s Secret volume / `docker run -v`, pointed at by
# API_KEYS_PATH) or, when no file exists, from the API_KEYS_JSON ENVIRONMENT VARIABLE (the same JSON as api_keys.json, e.g. a k8s Secret
# exposed with secretKeyRef). Nothing is ever baked into the image. If authentication is required and neither source yields a valid
# key store, the app does NOT crash: it logs a StartupError, keeps running, answers 503 on the prediction endpoints and reports
# auth_ready=false (HTTP 503) on /health until a key store is provided.
class StartupError(RuntimeError):
    """A mandatory runtime secret is missing. Created and LOGGED at start-up (see _startup_auth_check), never raised at import."""


_AUTH_STORE_MSG = "Authentication enabled but no valid key-store or env found at runtime."
_API_KEYS_ENV_RAW = os.environ.get("API_KEYS_JSON", "").strip()
_env_keys_cache = [None]


def _keys_from_env():
    """Key dict parsed from API_KEYS_JSON (empty when unset). Accepts the api_keys.json layout (an object with a "keys" object) or a
    bare object mapping key ids to entries. The error text never contains the variable's content."""
    if not _API_KEYS_ENV_RAW:
        return {}
    if _env_keys_cache[0] is None:
        try:
            _data = json.loads(_API_KEYS_ENV_RAW)
        except ValueError:
            raise ValueError("API_KEYS_JSON is not valid JSON")
        _keys = _data.get("keys", _data) if isinstance(_data, dict) else None
        if not isinstance(_keys, dict):
            raise ValueError("API_KEYS_JSON has no valid 'keys' object")
        _env_keys_cache[0] = _keys
    return _env_keys_cache[0]


def _api_keys_file_has_keys():
    if not (API_KEYS_PATH and os.path.isfile(API_KEYS_PATH)):
        if _API_KEYS_ENV_RAW:
            try:
                return bool(_keys_from_env())
            except ValueError:
                print("[Startup] WARNING: API_KEYS_JSON is set but is not valid JSON -- authentication stays ENFORCED and every "
                      "request gets 503 until it is fixed.")
                return True
        return False
    try:
        with open(API_KEYS_PATH, "r", encoding="utf-8") as f:
            return bool(json.load(f).get("keys"))
    except (OSError, ValueError, AttributeError):
        # [CORRUPT api_keys.json FIX] a file that EXISTS but cannot be parsed must never silently switch authentication
        # off: treat it as "keys configured" so auth stays enforced (require_api_key then answers a clear 503).
        print("[Startup] WARNING: " + str(API_KEYS_PATH) + " exists but is unreadable / not valid JSON -- authentication "
              "stays ENFORCED and every request gets 503 until the file is fixed or re-provisioned.")
        return True


def _resolve_require_auth():
    if _REQUIRE_AUTH_RAW in ("false", "0", "no", "off"):
        return False
    if _REQUIRE_AUTH_RAW in ("true", "1", "yes", "on"):
        return True
    if _api_keys_file_has_keys():
        return True
    if _APP_ENV in ("production", "prod"):
        return True
    print("[Startup] WARNING: no api_keys.json (or no keys in it) found at "
          + (API_KEYS_PATH or "<bundle dir>/api_keys.json")
          + " -- running with authentication DISABLED. Set REQUIRE_AUTH=true and provision a key before exposing "
            "this service beyond local/staging use, or set APP_ENV=production to refuse to start open.")
    return False


REQUIRE_AUTH = _resolve_require_auth()
if not REQUIRE_AUTH and _APP_ENV in ("production", "prod") and _REQUIRE_AUTH_RAW in ("false", "0", "no", "off"):
    print("[Startup] WARNING: APP_ENV=production with REQUIRE_AUTH=false -- the prediction endpoints are OPEN to every caller. "
          "Only do this behind a gateway that authenticates requests.")


# [KEY-FILE CACHE] api_keys.json used to be opened + parsed on EVERY request -- including every unauthenticated guess, so a flood of bad
# keys was a cheap disk/CPU DoS. The parsed file is now cached and re-read only when its (mtime, size) changes, so a rotated / revoked key
# (or a k8s Secret update, which swaps the file) still takes effect on the next request.
_api_keys_cache = [None, None]   # [(mtime_ns, size), parsed keys dict]
_api_keys_cache_lock = threading.Lock()


def _load_api_keys():
    if not (API_KEYS_PATH and os.path.isfile(API_KEYS_PATH)):
        return _keys_from_env()          # no mounted file: the API_KEYS_JSON environment variable (or {} when unset)
    try:
        _st = os.stat(API_KEYS_PATH)
        _sig = (_st.st_mtime_ns, _st.st_size)
        with _api_keys_cache_lock:
            if _api_keys_cache[0] == _sig and _api_keys_cache[1] is not None:
                return _api_keys_cache[1]
        with open(API_KEYS_PATH, "r", encoding="utf-8") as f:
            keys = json.load(f).get("keys", {})
    except (OSError, ValueError, AttributeError) as e:
        raise ValueError("api_keys.json is unreadable or not valid JSON: %s" % e)
    if not isinstance(keys, dict):
        raise ValueError("api_keys.json has no valid 'keys' object")
    with _api_keys_cache_lock:
        _api_keys_cache[0], _api_keys_cache[1] = _sig, keys
    return keys


_PREDICT_ROLES = frozenset(r.strip().lower() for r in os.environ.get("PREDICT_ROLES", "reader,predictor,admin").split(",") if r.strip())

# [BRUTE-FORCE / DoS GUARD] failed authentications used to be rejected BEFORE the rate limiter ever ran, so nothing throttled guessing.
# Each client IP may now fail AUTH_FAIL_LIMIT_PER_MINUTE (default 20) times a minute; further failures get 429 (and Retry-After) instead of
# 401 until the window clears (0 disables). Only 401s count -- a valid key with the wrong role (403) or a server-side problem (503) is not
# the caller guessing. A VALID key is never blocked by this guard: behind a proxy several clients may share one IP, and an attacker must
# not be able to lock the legitimate ones out by failing on purpose.
_AUTH_FAIL_LIMIT = _env_int("AUTH_FAIL_LIMIT_PER_MINUTE", 20)
_auth_fail_lock = threading.Lock()
_auth_fail_windows = {}   # client ip -> collections.deque of failure timestamps within the last 60 s
_DUMMY_SALT = "0" * 32


def _auth_fail_record(ip):
    """Records one failed attempt; returns True when this IP is now over its failure budget."""
    if _AUTH_FAIL_LIMIT <= 0:
        return False
    now = time.time()
    with _auth_fail_lock:
        if len(_auth_fail_windows) > 50000:
            for _k in [k for k, b in _auth_fail_windows.items() if not b or b[-1] < now - 60.0]:
                del _auth_fail_windows[_k]
            if len(_auth_fail_windows) > 50000:
                _auth_fail_windows.clear()      # attacker-controlled keys: reset rather than grow without bound
        bucket = _auth_fail_windows.setdefault(ip, collections.deque())
        while bucket and bucket[0] < now - 60.0:
            bucket.popleft()
        bucket.append(now)
        return len(bucket) > _AUTH_FAIL_LIMIT


def _verify_api_key(x_api_key):
    if not x_api_key:
        raise HTTPException(401, "Missing X-API-Key header.")
    key_id, _sep, raw_key = str(x_api_key).partition(".")
    try:
        entry = _load_api_keys().get(key_id) if _sep else None
    except ValueError as e:
        # the reason (parser message) goes to the pod log, never to an unauthenticated caller
        print("[Auth] api_keys.json cannot be read: " + str(e))
        raise HTTPException(503, "Authentication is enabled but the key store cannot be read. Check the server log.")
    # [UNIFORM FAILURE] unknown id, malformed key, revoked key and wrong secret all give the SAME 401 text and do the SAME work (one
    # HMAC, against a dummy salt when the id does not exist), so neither the message nor the response time tells a caller which key ids exist.
    _known = isinstance(entry, dict)
    _salt = str(entry.get("salt")) if _known and entry.get("salt") is not None else _DUMMY_SALT
    expected_hash = hmac.new(_salt.encode("utf-8"), (raw_key if _sep else str(x_api_key)).encode("utf-8"), hashlib.sha256).hexdigest()
    try:
        hash_ok = hmac.compare_digest(expected_hash, str(entry.get("hash")) if _known else "0" * 64)
    except (TypeError, ValueError):
        hash_ok = False
    if not (_known and hash_ok and not entry.get("revoked")):
        raise HTTPException(401, "Invalid API key.")
    try:
        _expired = entry.get("expires_at") is not None and time.time() > float(entry["expires_at"])
    except (TypeError, ValueError):
        _expired = True
    if _expired:
        raise HTTPException(401, "API key expired.")
    # [ROLE ENFORCEMENT] the role stored in api_keys.json used to be recorded for audit only: ANY valid key could call the endpoints.
    # Only roles listed in PREDICT_ROLES (default reader,predictor,admin) may score; a key with any other role (e.g. "disabled",
    # "billing") gets 403. With auth OFF the caller is "anonymous" (never "admin").
    _role = str(entry.get("role") or "").strip().lower()
    if _role not in _PREDICT_ROLES:
        raise HTTPException(403, "API key role '" + _role[:30] + "' is not allowed to call the prediction endpoints.")
    return {"key_id": key_id, "owner": entry.get("owner"), "role": _role}


def _auth_store_ok():
    """True when at least one key can be read from the mounted file or from API_KEYS_JSON."""
    try:
        return bool(_load_api_keys())
    except ValueError:
        return False


def _auth_ready():
    """Authentication is either not required, or its key store is present and valid (re-checked on every call: cheap, cached by mtime)."""
    return (not REQUIRE_AUTH) or _auth_store_ok()


def _startup_auth_check():
    """[RULE 2] Logs a StartupError -- and returns it -- when auth is required but no valid key store exists. Never raises."""
    if not REQUIRE_AUTH or _auth_store_ok():
        return None
    _err = StartupError(_AUTH_STORE_MSG)
    print("[StartupError] " + str(_err))
    print("[StartupError] Looked for api_keys.json at " + (API_KEYS_PATH or "<bundle dir>/api_keys.json")
          + (" (API_KEYS_PATH=" + _API_KEYS_PATH_ENV + ")" if _API_KEYS_PATH_ENV else "") + " and for the API_KEYS_JSON environment variable. "
          "The service keeps running: prediction endpoints answer 503 and /health reports auth_ready=false until a key store is mounted "
          "(Kubernetes Secret volume / docker -v) or API_KEYS_JSON is set.")
    return _err


def require_api_key(x_api_key: str = Header(default=None), request: Request = None):
    """FastAPI dependency -- add `identity: dict = Depends(require_api_key)`
    to any endpoint that must be authenticated."""
    if not REQUIRE_AUTH:
        return {"key_id": None, "owner": "anonymous", "role": "anonymous"}
    if not ((API_KEYS_PATH and os.path.isfile(API_KEYS_PATH)) or _API_KEYS_ENV_RAW):
        raise HTTPException(503, _AUTH_STORE_MSG)
    try:
        return _verify_api_key(x_api_key)
    except HTTPException as _e:
        if _e.status_code == 401 and _auth_fail_record(_client_ip(request) if request is not None else "unknown"):
            raise HTTPException(429, "Too many failed authentication attempts. Retry later.", headers={"Retry-After": "60"})
        raise


_AUTH_STARTUP_ERROR = _startup_auth_check()

# ---- Rate limiting (per API key / per IP) -------------------------------
# GAP FIX: nothing in the served API throttled request volume -- one
# noisy or abusive client could exhaust the process. This adds an
# in-memory sliding-window limiter keyed by the authenticated key_id
# (falls back to caller IP when REQUIRE_AUTH=false), so a single caller
# can't starve everyone else sharing this process.
# LIMITATION, stated plainly rather than silently ignored (the shipped k8s-ingress.yaml adds a cluster-wide limit at the ingress to cover it):
# the window is per-process. Behind multiple uvicorn/gunicorn workers or replicas, each
# process throttles independently, so the effective ceiling is
# RATE_LIMIT_PER_MINUTE * worker_count, not RATE_LIMIT_PER_MINUTE. For a
# multi-process deployment, swap _rate_limit_windows for a shared store
# (e.g. Redis INCR + EXPIRE) behind the same check_rate_limit() interface.
RATE_LIMIT_PER_MINUTE = _env_int("RATE_LIMIT_PER_MINUTE", 60)
_rate_limit_lock = threading.Lock()
_rate_limit_windows = {}  # caller key -> collections.deque of timestamps within the last 60s (popleft is O(1); list.pop(0) was O(n))
_rate_limit_last_purge = [0.0]
_RATE_LIMIT_MAX_BUCKETS = 50000   # hard cap on distinct callers kept in memory (purged early when exceeded)
# [PROXY-AWARE IP] behind an ingress / load balancer request.client.host is the PROXY's address, so with auth off every caller shared one
# bucket. TRUSTED_PROXY_HOPS=N (default 0 = ignore X-Forwarded-For, spoof-proof) takes the N-th address from the RIGHT of X-Forwarded-For,
# i.e. the one written by your own outermost trusted proxy. Set it to 1 behind a single ingress controller.
_TRUSTED_PROXY_HOPS = _env_int("TRUSTED_PROXY_HOPS", 0)


def _client_ip(request: Request) -> str:
    if _TRUSTED_PROXY_HOPS > 0:
        _xff = [p.strip() for p in request.headers.get("x-forwarded-for", "").split(",") if p.strip()]
        if len(_xff) >= _TRUSTED_PROXY_HOPS:
            return _xff[-_TRUSTED_PROXY_HOPS][:64]
    return request.client.host if request.client else "unknown"


def _rate_limit_caller_key(identity: dict, request: Request) -> str:
    if identity.get("key_id"):
        return f"key:{identity['key_id']}"
    return f"ip:{_client_ip(request)}"


def _rate_limit_check(request, identity):
    """The rate limiter proper (see check_rate_limit). Set RATE_LIMIT_PER_MINUTE=0 to disable throttling entirely."""
    if RATE_LIMIT_PER_MINUTE <= 0:
        return identity
    caller_key = _rate_limit_caller_key(identity, request)
    now = time.time()
    window_start = now - 60.0
    with _rate_limit_lock:
        # [MEMORY FIX] with auth off every client IP gets its own bucket and nothing ever removed them -> unbounded growth.
        # Once a minute, drop buckets whose newest request is older than the window.
        if now - _rate_limit_last_purge[0] > 60.0 or len(_rate_limit_windows) > _RATE_LIMIT_MAX_BUCKETS:
            _rate_limit_last_purge[0] = now
            for _k in [k for k, b in _rate_limit_windows.items() if not b or b[-1] < window_start]:
                del _rate_limit_windows[_k]
        bucket = _rate_limit_windows.get(caller_key)
        if bucket is None:
            bucket = _rate_limit_windows[caller_key] = collections.deque()
        while bucket and bucket[0] < window_start:
            bucket.popleft()
        if len(bucket) >= RATE_LIMIT_PER_MINUTE:
            retry_after = max(1, int(bucket[0] + 60.0 - now))
            raise HTTPException(
                429,
                f"Rate limit exceeded ({RATE_LIMIT_PER_MINUTE} requests/min). Retry later.",
                headers={"Retry-After": str(retry_after)},
            )
        bucket.append(now)
    return identity


def check_rate_limit(request: Request, identity: dict = Depends(require_api_key), background_tasks: BackgroundTasks = None):
    """FastAPI dependency -- use in place of `Depends(require_api_key)` on any endpoint that should be both authenticated AND
    rate-limited (it depends on require_api_key itself, so auth still runs first and a 401 still takes priority over a 429).
    [RULE 4] It also stamps the request start time and queues the monitor (latency window + drift) as a BackgroundTask, which runs
    AFTER the response has been sent -- the request path pays for one perf_counter() call."""
    _rate_limit_check(request, identity)
    _st = getattr(request, "state", None)
    if _st is not None:
        _st.monitor_t0 = time.perf_counter()
    if background_tasks is not None:
        background_tasks.add_task(_observe_request, request)
    return identity


class TabularRawRequest(BaseModel):
    features: dict  # {"feature_name": value, ...} -- raw, unprocessed values


try:
    import onnxruntime as ort
except ImportError:
    ort = None

ONNX_MODEL_PATH = _resolve_bundle_path("", "*.onnx")

# [RULE 3] the session is created by _load_onnx_session() (core block): safe try/except, external-data (.onnx.data) name repair that also
# works on a read-only root filesystem, explicit providers. It never raises -- a bad model file leaves _session None and /health reports it.
_session = None
_session_load_error = None
if ONNX_MODEL_PATH and os.path.isfile(ONNX_MODEL_PATH):
    _session, _session_load_error = _load_onnx_session(ort, ONNX_MODEL_PATH)
    if _session_load_error:
        print("[Startup] WARNING: failed to load ONNX model " + os.path.basename(ONNX_MODEL_PATH) + ": " + _session_load_error)


LINEAR_MODEL_JSON_PATH = _resolve_bundle_path("", "tabular_model.json")

_linear_model = _load_json_artifact(LINEAR_MODEL_JSON_PATH, "tabular_model.json")


def _predict_linear_json(x, model_payload):
    """Pure-NumPy inference for a bundler.export_classical_json model -- no sklearn needed.
    [TASK-TYPE FIX] binary logistic -> sigmoid, multinomial logistic -> softmax over every class row of coef_
    (the old code only ever looked at logits[0]); regressors return one float. A payload that carries classes_
    is a classifier whatever else it says; otherwise its "task"/the recipe's task decides."""
    coef = np.atleast_2d(np.asarray(model_payload["coef_"], dtype=np.float64))
    intercept = np.atleast_1d(np.asarray(model_payload["intercept_"], dtype=np.float64))
    scores = (x @ coef.T + intercept).ravel()
    task = _model_task(model_payload)
    is_classifier = (model_payload.get("classes_") is not None or model_payload["model_class"] == "LogisticRegression"
                     or task in ("binary", "multiclass", "classification"))
    if not is_classifier:
        return {"prediction": float(scores[0])}
    classes = model_payload.get("classes_")
    probs = [1.0 - float(_sigmoid(scores[0])), float(_sigmoid(scores[0]))] if scores.size == 1 else _softmax(scores)
    return _classification_result(probs, classes, require_threshold=True)


TREE_ENSEMBLE_JSON_PATH = _resolve_bundle_path("tabular_tree_ensemble.json", "tabular_tree_ensemble.json")

_tree_ensemble = _load_json_artifact(TREE_ENSEMBLE_JSON_PATH, "tabular_tree_ensemble.json")


def _traverse_tree(node, x_row):
    """Pure-NumPy binary-tree walk for one LightGBM/XGBoost-exported
    tree, already converted into tree_inference.py's schema by
    gbm_engine.export_tree_ensemble_to_json (see that function's
    bug-fix note for why a native dump can't be traversed directly).
    Respects each node's own "op" ('<=' for LightGBM, '<' for
    XGBoost -- defaults to '<=' for old exports predating this field)
    instead of assuming '<=' everywhere."""
    while "leaf_value" not in node:
        threshold = node["threshold"]
        op = node.get("op", "<=")
        feature_val = x_row[node["split_feature"]]
        if op == "<":
            # [XGBOOST FLOAT32 FIX] XGBoost compares in float32; the row is rounded to float32 precision by the caller and
            # the threshold the same way (cached on the node), so values sitting exactly ON a split go the same side as in
            # the trained model (before: probabilities off by up to 0.88, recall ~10 points lower than reported).
            t32 = node.get("_t32")
            if t32 is None:
                t32 = node["_t32"] = float(np.float32(threshold))
            goes_left = feature_val < t32
        else:
            goes_left = feature_val <= threshold
        node = node["left"] if goes_left else node["right"]
    return node["leaf_value"]


def _predict_tree_ensemble_json(x, ensemble_payload):
    """Sums every tree's leaf contribution (standard gradient-boosting inference) and turns the raw score into the
    right output FOR THE MODEL'S TASK -- bypasses sklearn/lightgbm/xgboost at serving time.
    [TASK-TYPE FIX] export_tree_ensemble_to_json now records task / num_class / base_score. The old code defaulted
    to "classification" whenever the key was missing, so every regression ensemble got a sigmoid and returned 0/1.
    Resolution order: payload["task"] -> feature_recipe.json["task"] -> payload["objective"]; only if all three are
    silent does it fall back to the legacy binary-classification behaviour."""
    trees = ensemble_payload["trees"]
    base_score = ensemble_payload.get("base_score", 0.0)
    learning_rate = ensemble_payload.get("learning_rate", 1.0)
    num_class = int(ensemble_payload.get("num_class", 1) or 1)
    x_row = x[0]
    if str(ensemble_payload.get("engine", "")).lower() == "xgboost" or (trees and str(trees[0].get("op", "")) == "<"):
        x_row = np.asarray(x_row, dtype=np.float32).astype(np.float64)   # [XGBOOST FLOAT32 FIX]
    if num_class > 1:
        raw = np.zeros(num_class, dtype=np.float64)
        for i, tree in enumerate(trees):  # boosting rounds interleave one tree per class
            raw[i % num_class] += learning_rate * _traverse_tree(tree, x_row)
        return _classification_result(_softmax(raw), ensemble_payload.get("classes_"))
    raw_score = float(base_score)
    for tree in trees:
        raw_score += learning_rate * _traverse_tree(tree, x_row)
    task = _model_task(ensemble_payload)
    if not task:
        objective = str(ensemble_payload.get("objective", "")).lower()
        task = ("binary" if "binary" in objective or "logistic" in objective else
                "regression" if objective else "")
    if not task:
        # [NO SILENT GUESS] a regressor pushed through a sigmoid returns plausible-looking 0/1 values -- fail loudly.
        raise HTTPException(500, "Cannot tell whether this tree ensemble is a regressor or a classifier (no task / "
                                 "objective in the artifact, model_task.json or feature_recipe.json). Set the "
                                 "MODEL_TASK_OVERRIDE env var to regression | binary | multiclass, or re-export the model.")
    if task == "regression":
        return {"prediction": float(raw_score)}
    prob = float(_sigmoid(raw_score))
    return _classification_result([1.0 - prob, prob], ensemble_payload.get("classes_"), require_threshold=True)


FEATURE_RECIPE_PATH = _resolve_bundle_path("feature_recipe.json", ("feature_recipe.json", "ann_fallback_recipe.json"))

_recipe = _load_json_artifact(FEATURE_RECIPE_PATH, "feature_recipe.json")


# ---- Task-type resolution (regression | binary | multiclass) -----------------------------------------------------
# [TASK-TYPE FIX] The serving code used to decide "classification vs regression" from a default string
# ("classification") when the artifact did not say -- so a regression LightGBM/XGBoost ensemble was pushed through
# a sigmoid and returned 0/1. The task is now read from the model artifact first, then feature_recipe.json, and the
# prediction shape follows it: regression -> a float; binary/multiclass -> class label + probabilities.
_REGRESSION_TASKS = ("regression", "regressor", "reg", "timeseries", "time_series", "forecast", "forecasting")


def _normalize_task(raw):
    t = str(raw or "").strip().lower().replace("-", "_")
    if t in _REGRESSION_TASKS:
        return "regression"
    if t in ("multiclass", "multi_class", "multiclass_classification"):
        return "multiclass"
    if t in ("binary", "binary_classification"):
        return "binary"
    if t in ("classification", "classifier", "clf"):
        return "classification"
    return ""


_MODEL_TASK_OVERRIDE = _normalize_task(os.environ.get("MODEL_TASK_OVERRIDE", ""))
if os.environ.get("MODEL_TASK_OVERRIDE", "").strip() and not _MODEL_TASK_OVERRIDE:
    print("[Startup] WARNING: MODEL_TASK_OVERRIDE='" + os.environ.get("MODEL_TASK_OVERRIDE", "") + "' is not one of "
          "regression | binary | multiclass | classification -- ignored.")

_model_task_file = None
_MODEL_TASK_FILE = os.path.join(_BUNDLE_DIR, "model_task.json")
if os.path.isfile(_MODEL_TASK_FILE):
    try:
        with open(_MODEL_TASK_FILE, "r", encoding="utf-8") as _f_task:
            _model_task_file = json.load(_f_task)
    except (OSError, ValueError) as _task_err:
        print("[Startup] WARNING: model_task.json exists but is unreadable (" + str(_task_err) + ") -- ignored.")


def _model_task(payload=None):
    """Task of the loaded model, resolved in this order: MODEL_TASK_OVERRIDE env var -> the model artifact's own
    "task" key -> model_task.json (bundle dir) -> feature_recipe.json's "task". "" if none of them says."""
    if _MODEL_TASK_OVERRIDE:
        return _MODEL_TASK_OVERRIDE
    for src in (payload, _model_task_file, globals().get("_recipe")):  # some bundles have no _recipe
        if isinstance(src, dict):
            t = _normalize_task(src.get("task"))
            if t:
                return t
    return ""


def _recipe_class_names():
    rec = globals().get("_recipe")
    names = rec.get("class_names") if isinstance(rec, dict) else None
    return list(names) if names else None


def _softmax(z):
    z = np.asarray(z, dtype=np.float64).ravel()
    e = np.exp(z - z.max())
    return e / e.sum()


def _sigmoid(v):
    return 1.0 / (1.0 + np.exp(-np.clip(v, -500.0, 500.0)))


_THR_WARNED = []   # warn about an unusable recipe threshold ONCE, not on every request


def _decision_threshold_cfg():
    """The manual decision threshold this model was BUILT with, from feature_recipe.json: `decision_threshold`
    (TABULAR_ML, tabular_ml_settings.decision_threshold) or `ann_decision_threshold` (ANN, ann_imbalance_settings).
    A dict {threshold, positive_index} (a bare float is accepted too). For a bundle built WITHOUT one the deployer can set
    the DECISION_THRESHOLD env var by hand. None -> no threshold known: binary tabular/ANN scoring then refuses to guess a
    0.50 cut (see _threshold_missing_error). Out-of-range values are ignored with a warning."""
    _rec = globals().get("_recipe")
    _dt = None
    if isinstance(_rec, dict):
        _dt = _rec.get("decision_threshold")
        if _dt is None:
            _dt = _rec.get("ann_decision_threshold")
    if str(os.environ.get("DECISION_THRESHOLD", "")).strip():
        # [THRESHOLD OVERRIDE] the deployer's DECISION_THRESHOLD env var wins over the value baked into the recipe, so the
        # precision/recall trade-off can be tuned per deployment (kubectl set env) without retraining. Keeps the recipe's
        # positive_index. Unset the variable to go back to the trained threshold.
        _env_thr = str(os.environ["DECISION_THRESHOLD"]).strip()
        _pos_idx = int(_dt.get("positive_index", 1)) if isinstance(_dt, dict) else 1
        _dt = {"threshold": _env_thr, "positive_index": _pos_idx}
    if isinstance(_dt, dict) and isinstance(_dt.get("threshold"), str):
        try:
            _dt = {**_dt, "threshold": float(_dt["threshold"])}
        except ValueError:
            return None
    if isinstance(_dt, str):
        try:
            _dt = float(_dt)
        except ValueError:
            return None
    if isinstance(_dt, (int, float)) and not isinstance(_dt, bool):
        _dt = {"threshold": float(_dt), "positive_index": 1}
    if not isinstance(_dt, dict) or _dt.get("threshold") is None:
        return None
    try:
        _thr = float(_dt["threshold"])
        _pos = int(_dt.get("positive_index", 1))
    except (TypeError, ValueError):
        return None
    if not (0.0 < _thr < 1.0) or _pos not in (0, 1):
        if not _THR_WARNED:
            _THR_WARNED.append(1)
            print("[Startup] WARNING: decision_threshold in feature_recipe.json is not a float in (0,1) -- ignored.")
        return None
    return {"threshold": _thr, "positive_index": _pos}


def _threshold_class_index(probs):
    """Predicted class INDEX for a binary probability vector under the manual threshold; None when no threshold is set
    or the vector is not binary (then callers keep their normal argmax)."""
    _dt = _decision_threshold_cfg()
    probs = np.asarray(probs, dtype=np.float64).ravel()
    if _dt is None or probs.size != 2:
        return None
    _pos = _dt["positive_index"]
    return _pos if float(probs[_pos]) >= _dt["threshold"] else 1 - _pos


def _is_binary_classifier():
    """True when the loaded model scores exactly two classes -- decided from the recipe, else the model artifact itself (the
    recipe's class_names can be null), so the check works for every backend."""
    rec = globals().get("_recipe")
    if isinstance(rec, dict) and str(rec.get("task", "")).lower() in ("regression", "regressor"):
        return False
    names = _recipe_class_names()
    if names:
        return len(names) == 2
    for _m in (globals().get("_tree_ensemble"), globals().get("_linear_model")):
        if isinstance(_m, dict):
            if _m.get("classes_"):
                return len(_m["classes_"]) == 2
            if str(_m.get("task", "")).lower() == "binary":
                return True
    sess = globals().get("_session")
    if sess is not None:
        try:
            return any(len(o.shape or []) == 2 and o.shape[1] == 2 for o in sess.get_outputs())
        except Exception:  # noqa: BLE001
            return False
    return False


def _threshold_required():
    """Tabular / ANN binary classifiers have NO default cut-off (the automatic 0.50 was removed on purpose): one must come from
    feature_recipe.json or the DECISION_THRESHOLD env var. CNN / RNN apps share this fragment but have no threshold concept."""
    return MODEL_TYPE in ("TABULAR_ML", "ANN") and _is_binary_classifier()


def _threshold_ready():
    return (not _threshold_required()) or _decision_threshold_cfg() is not None


def _threshold_missing_error():
    return HTTPException(500, "This binary classifier has no decision threshold: feature_recipe.json carries none and the "
                              "DECISION_THRESHOLD env var is not set. The automatic 0.50 default was removed -- re-train with a "
                              "manual decision_threshold, or set DECISION_THRESHOLD (a number strictly between 0 and 1, e.g. "
                              "0.35) for this deployment.")


def _classification_result(probs, classes=None, require_threshold=False):
    """probs: 1-D class probabilities. `prediction` is the model's own class label when known (classes_), else the
    class index; `class_name` adds the human-readable name from the recipe when it differs."""
    probs = np.asarray(probs, dtype=np.float64).ravel()
    idx = int(np.argmax(probs))
    _dt = _decision_threshold_cfg()
    _t_idx = _threshold_class_index(probs)
    if _t_idx is None and require_threshold and probs.size == 2:   # tabular/ANN binary: no silent 0.50 / argmax cut
        raise _threshold_missing_error()
    if _t_idx is not None:  # binary model built with a manual decision threshold -> the SAME cut-off at serving
        idx = _t_idx
    label = classes[idx] if classes and idx < len(classes) else idx
    if isinstance(label, np.generic):
        label = label.item()
    if isinstance(label, float) and label.is_integer():   # classes_ stored as 0.0 / 1.0 -> answer 0 / 1
        label = int(label)
    result = {"prediction": label, "probability": float(probs[idx]), "probabilities": probs.tolist()}
    names = _recipe_class_names()
    if names and idx < len(names) and names[idx] != label:
        result["class_name"] = names[idx]
    if _dt is not None and probs.size == 2:
        result["decision_threshold"] = float(_dt["threshold"])
    return result


def _with_class_probs(result, raw_output):
    """Image / sequence-text classifiers used to return only the raw logits under `prediction`. That field is left exactly as
    it was (no contract break) and the decoded answer is ADDED: predicted_class + probabilities."""
    cls = _postprocess_onnx_tabular([np.asarray(raw_output)], "classification", require_threshold=False)
    result["predicted_class"] = cls["prediction"]
    result["probabilities"] = cls["probabilities"]
    return result


def _postprocess_onnx_tabular(outputs, task, require_threshold=True):
    """Shape an ONNX Runtime result by task. Regression keeps the old behaviour (raw output as `prediction`).
    Classification: skl2onnx classifiers return [labels, probabilities]; torch MLPs return one logits (or
    sigmoid/softmax) tensor -- both become {prediction, probability, probabilities}."""
    if task in ("regression", ""):  # "" = artifact + recipe silent -> legacy pass-through, never guess classification
        return {"prediction": np.asarray(outputs[0]).tolist()}
    if len(outputs) > 1:
        label = np.asarray(outputs[0]).ravel()[0]
        label = label.item() if isinstance(label, np.generic) else label
        raw = outputs[1]
        if isinstance(raw, list) and raw and isinstance(raw[0], dict):
            keys = list(raw[0].keys())
            probs = [float(raw[0][k]) for k in keys]
            res = {"prediction": label, "probability": float(max(probs)), "probabilities": probs}
            res["classes"] = [k.item() if isinstance(k, np.generic) else k for k in keys]
            _t_idx = _threshold_class_index(probs)
            if _t_idx is None and require_threshold and len(probs) == 2:
                raise _threshold_missing_error()
            if _t_idx is not None:   # manual threshold: the ONNX label is the default-0.50 label -> re-decide from probs
                _lab = keys[_t_idx]
                res["prediction"] = _lab.item() if isinstance(_lab, np.generic) else _lab
                res["probability"] = float(probs[_t_idx])
                res["decision_threshold"] = _decision_threshold_cfg()["threshold"]
            return res
        probs = np.asarray(raw, dtype=np.float64).reshape(-1)
        res = _classification_result(probs, require_threshold=require_threshold)
        if _threshold_class_index(probs) is None:
            res["prediction"] = label       # no manual threshold: ONNX's own label (default cut) as before
        # with a manual threshold _classification_result() already decided the label from the probabilities; targets are
        # label-encoded 0..k-1, so the class index IS the label the ONNX graph would have emitted.
        return res
    vec = np.asarray(outputs[0], dtype=np.float64).reshape(-1)
    if vec.size == 1:
        p = float(vec[0])
        if not 0.0 <= p <= 1.0:
            p = float(_sigmoid(p))
        return _classification_result([1.0 - p, p], require_threshold=require_threshold)
    if not (np.all(vec >= 0.0) and np.all(vec <= 1.0) and abs(float(vec.sum()) - 1.0) < 1e-3):
        vec = _softmax(vec)
    return _classification_result(vec, require_threshold=require_threshold)


def _encode_one_hot(value, categories):
    """Maps a raw category string to its exact one-hot position within
    the STATIC category array captured at training time. Unknown values
    map to an all-zero vector -- matches sklearn's OneHotEncoder
    handle_unknown='ignore' behavior exactly."""
    vec = [0.0] * len(categories)
    if value in categories:
        vec[categories.index(value)] = 1.0
    elif value is not None and str(value) in categories:  # JSON sent 1 where training saw "1"
        vec[categories.index(str(value))] = 1.0
    return vec


_DERIVED_DATE_FORMATS = ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d", "%d-%m-%Y", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S",
                         "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S",
                         "%Y/%m/%d %H:%M", "%d-%m-%Y %H:%M:%S", "%d-%m-%Y %H:%M", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M")


def _parse_date_value(value):
    """Same format list (and order) the training pipeline used to expand date columns; None when unparseable."""
    import datetime as _dt
    if isinstance(value, _dt.datetime):
        return value
    if isinstance(value, _dt.date):
        return _dt.datetime(value.year, value.month, value.day)
    text = str(value).strip()
    for fmt in _DERIVED_DATE_FORMATS:
        try:
            return _dt.datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _apply_derived_features(features, recipe):
    """[RAW-INPUT CONTRACT] The model was trained on engineered columns (date parts, geo distance, frequency codes). The
    recipe records how each was built, so the client can send the RAW columns (e.g. trans_date_trans_time, dob, lat, long,
    merch_lat, merch_long, merchant) and the server derives the engineered ones. A value the client sends for the
    engineered column itself always wins. Returns a new dict; the request is not modified."""
    feats = dict(features or {})
    for spec in (recipe.get("derived_features") or []):
        kind = spec.get("type")
        if kind == "date_parts":
            raw = feats.get(spec["source"])
            if raw is None:
                if spec["source"] in feats:
                    # [NULL-INPUT FIX] an explicit null date means "unknown": its parts are unknown too and get imputed below,
                    # instead of being reported as missing features (422) -- only a column left out entirely is rejected.
                    for part in spec.get("parts", []):
                        feats.setdefault(spec["prefix"] + "_" + part, None)
                continue
            parsed = _parse_date_value(raw)
            if parsed is None:
                raise HTTPException(422, "Column '" + str(spec["source"]) + "' is not a date this model understands: "
                                         + repr(raw)[:60] + " (use e.g. 2020-09-03 23:36:17).")
            parts = {"year": parsed.year, "month": parsed.month, "day": parsed.day,
                     "dayofweek": parsed.weekday(), "hour": parsed.hour}
            for part in spec.get("parts", []):
                feats.setdefault(spec["prefix"] + "_" + part, parts[part])
                if feats[spec["prefix"] + "_" + part] is None:
                    feats[spec["prefix"] + "_" + part] = parts[part]
        elif kind == "geo_distance":
            out = spec["out"]
            if feats.get(out) is not None:
                continue
            _coord_keys = [spec[k] for k in ("lat1", "lon1", "lat2", "lon2")]
            if not all(k in feats for k in _coord_keys):
                continue          # coordinates not sent at all -> the required-feature check reports what is missing
            _coord_vals = [feats[k] for k in _coord_keys]
            if any(v is None or (isinstance(v, float) and v != v) for v in _coord_vals):
                # [NULL-INPUT FIX] explicit null coordinate = unknown distance (imputed below), not a "missing feature" 422
                feats.setdefault(out, None)
                continue
            try:
                a1, o1, a2, o2 = (float(v) for v in _coord_vals)
            except (TypeError, ValueError):
                raise HTTPException(422, "Coordinates " + ", ".join(_coord_keys) + " must be numeric.")
            if any(abs(v) == float("inf") for v in (a1, o1, a2, o2)):
                raise HTTPException(422, "Coordinates " + ", ".join(_coord_keys) + " must be finite numbers.")
            if any(v != v for v in (a1, o1, a2, o2)):    # "nan" text -> unknown distance, imputed below like a null
                feats.setdefault(out, None)
                continue
            import math as _m
            r1, r2 = _m.radians(a1), _m.radians(a2)
            h = _m.sin((r2 - r1) / 2) ** 2 + _m.cos(r1) * _m.cos(r2) * _m.sin(_m.radians(o2 - o1) / 2) ** 2
            feats[out] = 2 * 6371.0088 * _m.asin(_m.sqrt(min(max(h, 0.0), 1.0)))
        elif kind == "frequency":
            out = spec["out"]
            if feats.get(out) is not None:
                continue
            if feats.get(spec["source"]) is None:
                if spec["source"] in feats:
                    feats.setdefault(out, None)   # [NULL-INPUT FIX] explicit null source -> unknown code, imputed below
                continue
            feats[out] = float(spec["map"].get(str(feats[spec["source"]]), spec.get("unseen", 0.0)))
        elif kind == "log1p":
            # [FEATURE EXPANSION] log1p twin of a right-skewed numeric column: log1p(max(x, 0)), exactly as in training
            out = spec["out"]
            if feats.get(out) is not None:
                continue
            _src = feats.get(spec["source"])
            if _src is None:
                if spec["source"] in feats:
                    feats.setdefault(out, None)
                continue
            try:
                _sv = float(_src)
            except (TypeError, ValueError):
                continue          # the numeric loop below reports the bad SOURCE column with a clear 422
            if _sv != _sv:
                feats.setdefault(out, None)
                continue
            if _sv in (float("inf"), float("-inf")):
                continue          # the numeric loop below rejects non-finite source values
            import math as _m_l1p
            feats[out] = _m_l1p.log1p(max(_sv, 0.0))
        elif kind == "ordinal":
            # [FEATURE EXPANSION] ordinal ranking; a level never seen in training -> None -> imputed with the training median
            out = spec["out"]
            if feats.get(out) is not None:
                continue
            if spec["source"] not in feats:
                continue
            if feats.get(spec["source"]) is None:
                feats.setdefault(out, None)
                continue
            feats[out] = spec["map"].get(str(feats[spec["source"]]).strip().lower())
        elif kind == "category_map":
            # [FEATURE EXPANSION] smoothed target-mean / frequency code of a high-cardinality category; unseen / null -> prior
            out = spec["out"]
            if feats.get(out) is not None:
                continue
            if spec["source"] not in feats:
                continue
            _raw_cat = feats.get(spec["source"])
            _cat_key = "__missing__" if (_raw_cat is None or (isinstance(_raw_cat, float) and _raw_cat != _raw_cat)) else str(_raw_cat)
            feats[out] = float(spec["map"].get(_cat_key, spec.get("unseen", 0.0)))
    return feats


def _check_required_features(features, recipe):
    """[SILENT-IMPUTATION FIX] A request that simply OMITS a feature used to be scored anyway (the median was filled in) and
    answered 200 -- a client typo or a forgotten field produced a confident-looking fraud score. Omitted keys now give HTTP
    422 naming them. An explicit null still means "unknown" and is imputed. ALLOW_MISSING_FEATURES=true restores the old
    behaviour for a deployment that wants it."""
    if str(os.environ.get("ALLOW_MISSING_FEATURES", "")).strip().lower() in ("1", "true", "yes", "on"):
        return
    missing = [c for c in list(recipe.get("numeric_cols") or []) + list(recipe.get("categorical_cols") or [])
               if c not in features]
    if missing:
        # [MESSAGE FIX] engineered columns (dob_year, geo_dist_km, ...) are built from RAW columns the client actually sends;
        # naming the engineered ones sent callers looking for columns that do not exist in their data. Report the raw source.
        _raw_of = {}
        for _spec in (recipe.get("derived_features") or []):
            if _spec.get("type") == "date_parts":
                for _p in _spec.get("parts", []):
                    _raw_of[_spec["prefix"] + "_" + _p] = [_spec["source"]]
            elif _spec.get("type") == "geo_distance":
                _raw_of[_spec["out"]] = [_spec[k] for k in ("lat1", "lon1", "lat2", "lon2")]
            elif _spec.get("type") in ("frequency", "ordinal", "category_map", "log1p"):
                _raw_of[_spec["out"]] = [_spec["source"]]
        _shown = []
        for _c in missing:
            for _r in (_raw_of.get(_c) or [_c]):
                if _r not in features and _r not in _shown:
                    _shown.append(_r)
        missing = _shown or missing
        raise HTTPException(422, "Missing required feature(s): " + ", ".join(missing[:25])
                                 + (" ..." if len(missing) > 25 else "")
                                 + ". Send them (null = unknown), or set ALLOW_MISSING_FEATURES=true to impute silently.")



def _layer_a_preprocess_tabular(features, recipe, notes=None):
    """Replays feature_recipe.json's "applied_pipeline_steps" ledger IN
    THE EXACT RECORDED ORDER: impute (median) -> winsorize (outlier
    caps) -> scale (standard/minmax) -> one-hot categoricals -> final
    concat, matching preprocessing.auto_preprocess()'s training-time
    transform byte-for-byte. Pure NumPy + stdlib, no sklearn needed
    here."""
    features = _apply_derived_features(features, recipe)        # [RAW-INPUT CONTRACT]
    _check_required_features(features, recipe)                   # [SILENT-IMPUTATION FIX]
    numeric_cols = recipe["numeric_cols"]
    categorical_cols = recipe["categorical_cols"]

    numeric_vals = []
    _imputed_cols = []       # [WARNINGS] null / NaN inputs replaced by the training median -- reported back to the caller
    for i, col in enumerate(numeric_cols):
        raw = features.get(col, None)
        if raw is None or (isinstance(raw, float) and raw != raw):  # None or NaN
            _imputed_cols.append(str(col))
            raw = recipe["impute_medians"][i] if recipe.get("impute_medians") else 0.0
        try:
            _num = float(raw)
        except (TypeError, ValueError):
            raise HTTPException(422, "Feature '" + str(col) + "' must be numeric, got " + repr(raw)[:60])
        if _num != _num:          # [NON-FINITE FIX] the STRING "nan" slipped past the NaN check above and was scored as NaN
            _imputed_cols.append(str(col))
            _num = float(recipe["impute_medians"][i]) if recipe.get("impute_medians") else 0.0
        elif _num in (float("inf"), float("-inf")):
            raise HTTPException(422, "Feature '" + str(col) + "' must be a finite number, got " + repr(raw)[:60])
        numeric_vals.append(_num)
    x_numeric = np.array(numeric_vals, dtype=np.float64).reshape(1, -1)

    if recipe.get("outlier_bounds"):
        for i, bounds in enumerate(recipe["outlier_bounds"]):
            x_numeric[0, i] = min(max(x_numeric[0, i], bounds["lower"]), bounds["upper"])

    scale_recipe = recipe.get("scale_recipe", {"method": "none"})
    if scale_recipe["method"] == "standard":
        mean_ = np.array(scale_recipe["mean_"])
        scale_ = np.array(scale_recipe["scale_"])
        x_numeric = (x_numeric - mean_) / scale_
    elif scale_recipe["method"] == "minmax":
        data_min_ = np.array(scale_recipe["data_min_"])
        data_max_ = np.array(scale_recipe["data_max_"])
        rng = data_max_ - data_min_
        rng[rng == 0] = 1.0
        x_numeric = (x_numeric - data_min_) / rng

    cat_vec = []
    _unseen_cats = []        # [WARNINGS] a category never seen in training is encoded as all zeros -- reported, not silent
    if categorical_cols:
        onehot_categories = recipe["encode_recipe"]["encoders"].get("onehot_categories", [])
        for col, categories in zip(categorical_cols, onehot_categories):
            raw_val = features.get(col, None)
            if raw_val is None or (raw_val not in categories and str(raw_val) not in categories):
                _unseen_cats.append(str(col) + "=" + repr(raw_val)[:40])
            cat_vec.extend(_encode_one_hot(raw_val, categories))
    if notes is not None:
        if _imputed_cols:
            notes.append("Null/NaN numeric features were imputed with training medians: " + ", ".join(_imputed_cols))
        if _unseen_cats:
            notes.append("Unseen or null categories were encoded as all zeros: " + ", ".join(_unseen_cats))

    if cat_vec:
        x_full = np.hstack([x_numeric, np.array(cat_vec, dtype=np.float64).reshape(1, -1)])
    else:
        x_full = x_numeric
    return x_full


DRIFT_BASELINE_PATH = _resolve_bundle_path("drift_baseline.json", "drift_baseline.json")

_drift_baseline = _load_json_artifact(DRIFT_BASELINE_PATH, "drift_baseline.json")


# ---- Live drift monitoring: PSI vs drift_baseline.json -- evaluated in the BACKGROUND (see the monitor block in the core section) ----
# DRIFT_WINDOW_SIZE (default 200), DRIFT_MIN_WINDOW, DRIFT_EVAL_EVERY and DRIFT_PSI_THRESHOLD are read there. Nothing below runs inside
# the request-response path: the endpoint only hands the preprocessed row to a FastAPI BackgroundTask.


def _psi(expected, actual, bins=10):
    """Population Stability Index of one column. Binary / constant columns (one-hot features) are binned on their own
    values; continuous ones on the baseline's percentiles with the outer bins open-ended (so out-of-range values count).
    Bin proportions are Laplace-smoothed (+0.5) instead of the old 1e-6 floor, which turned a single empty bin of a small
    window into a PSI of several units."""
    expected = np.asarray(expected, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    uniq = np.unique(expected)
    if uniq.size <= 2:
        edges = np.concatenate(([-np.inf], (uniq[:-1] + uniq[1:]) / 2.0, [np.inf]))
    else:
        edges = np.unique(np.percentile(expected, np.linspace(0, 100, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
    e_cnt, _ = np.histogram(expected, bins=edges)
    a_cnt, _ = np.histogram(actual, bins=edges)
    k = len(e_cnt)
    e_pct = (e_cnt + 0.5) / (e_cnt.sum() + 0.5 * k)
    a_pct = (a_cnt + 0.5) / (a_cnt.sum() + 0.5 * k)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


_drift_baseline_arr = [None]


def _drift_baseline_array():
    """[PERF FIX] the baseline samples as ONE float64 array, built once. It used to be rebuilt from a list of lists on every
    single request (~10x the cost of scoring the row itself)."""
    if _drift_baseline_arr[0] is None:
        _drift_baseline_arr[0] = np.asarray(_drift_baseline["samples"], dtype=np.float64)
    return _drift_baseline_arr[0]


_drift_prepared = [None]


def _drift_feature_monitored(index, feature):
    """[CALENDAR-YEAR DRIFT FIX] False for a calendar-year feature: its training values (e.g. 2019 / 2020) cannot match live traffic, so its PSI
    is ~3 from the first request of a new year and the drift warning stayed on forever. Honours the baseline's own monitor flag; a baseline
    written before the flag existed falls back to the same rule (name year / *_year and at most 4 bins -> only a few distinct years)."""
    _flag = feature.get("monitor")
    if _flag is not None:
        return bool(_flag)
    _names = _drift_baseline.get("feature_names") or []
    _name = str(_names[index]).strip().lower() if index < len(_names) else ""
    return not ((_name == "year" or _name.endswith("_year")) and len(feature.get("counts") or []) <= 4)


def _drift_baseline_prepared():
    """[PERF FIX] per-feature bin edges + baseline proportions, computed ONCE. _psi() re-derived the baseline's unique values,
    percentiles and histogram for every feature on every request (about 9 of the ~12 ms a scored row cost). The maths is
    identical to _psi(): same edges, same Laplace smoothing."""
    if _drift_prepared[0] is None and "samples" not in _drift_baseline and _drift_baseline.get("features"):
        # [PRIVACY FIX] histogram_v1 baseline: per-feature inner bin edges + training counts. No individual training row (dob,
        # location, amount ...) is shipped in the image any more -- yet the bin edges / proportions below are identical to
        # what the old raw-sample baseline produced, so the PSI numbers do not change.
        prepared = []
        for _fi, _ft in enumerate(_drift_baseline["features"]):
            if not _drift_feature_monitored(_fi, _ft):
                prepared.append(None)     # [CALENDAR-YEAR DRIFT FIX] never part of the PSI
                continue
            _edges = np.concatenate(([-np.inf], np.asarray(_ft["edges"], dtype=np.float64), [np.inf]))
            _cnt = np.asarray(_ft["counts"], dtype=np.float64)
            prepared.append((_edges, (_cnt + 0.5) / (_cnt.sum() + 0.5 * len(_cnt)), len(_cnt)))
        _drift_prepared[0] = prepared
    if _drift_prepared[0] is None:
        baseline = _drift_baseline_array()
        prepared = []
        for i in range(baseline.shape[1]):
            expected = baseline[:, i]
            uniq = np.unique(expected)
            if uniq.size <= 2:
                edges = np.concatenate(([-np.inf], (uniq[:-1] + uniq[1:]) / 2.0, [np.inf]))
            else:
                edges = np.unique(np.percentile(expected, np.linspace(0, 100, 5 + 1)))
                edges[0], edges[-1] = -np.inf, np.inf
            e_cnt, _ = np.histogram(expected, bins=edges)
            k = len(e_cnt)
            prepared.append((edges, (e_cnt + 0.5) / (e_cnt.sum() + 0.5 * k), k))
        _drift_prepared[0] = prepared
    return _drift_prepared[0]


def _drift_evaluate(window_arr):
    """PSI of every feature of the sliding window against the baseline; returns (max_psi, index_of_the_worst_feature)."""
    prepared = _drift_baseline_prepared()
    n_features = min(len(prepared), window_arr.shape[1])
    max_psi, worst = 0.0, 0
    for i in range(n_features):
        if prepared[i] is None:      # calendar-year feature, not monitored
            continue
        edges, e_pct, k = prepared[i]
        a_cnt, _ = np.histogram(window_arr[:, i], bins=edges)
        a_pct = (a_cnt + 0.5) / (a_cnt.sum() + 0.5 * k)
        psi_i = float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))
        if psi_i > max_psi:
            max_psi, worst = psi_i, i
    return max_psi, worst


if _drift_baseline is not None:
    _drift_evaluator[0] = _drift_evaluate


def _score_tabular(features):
    """The actual CPU-bound scoring work -- run inside asyncio.to_thread
    by the endpoint below so the event loop is never blocked by it."""

    if _recipe is None:
        raise HTTPException(503, "feature_recipe.json not found -- cannot preprocess raw features.")
    _notes = []
    x = _layer_a_preprocess_tabular(features, _recipe, _notes)

    if _session is not None:
        input_name = _session.get_inputs()[0].name
        outputs = _session.run(None, {input_name: x.astype(np.float32)})
        result = _postprocess_onnx_tabular(outputs, _model_task())
    elif _linear_model is not None:
        result = _predict_linear_json(x, _linear_model)
    elif _tree_ensemble is not None:
        result = _predict_tree_ensemble_json(x, _tree_ensemble)
    else:
        raise HTTPException(503, "No model artifact loaded (checked ONNX, linear-JSON, tree-JSON paths).")

    # [HETEROSCEDASTICITY LOG TRANSFORM item] Mirrors orchestrator.
    # train_and_evaluate()'s inverse transform at serving time: if this
    # model was trained on log1p(y_train) (recorded in feature_recipe.json
    # by run_tabular_ml_auto -- see that function's docstring), every raw
    # prediction from ANY of the three paths above (ONNX/linear-JSON/
    # tree-ensemble-JSON) is still in log-space and must be expm1'd back
    # to the real, original scale before a caller ever sees it. Without
    # this, a deployed regression endpoint would silently return numbers
    # on the wrong scale (e.g. ~10.3 instead of ~30000) with no error --
    # the single most dangerous kind of bug, since nothing about the
    # response shape looks wrong.
    if _recipe.get("log_transform_target") and _model_task() not in ("binary", "multiclass", "classification"):
        result["prediction"] = np.expm1(np.asarray(result["prediction"], dtype=np.float64)).tolist()


    if _notes and isinstance(result, dict):
        result["warnings"] = _notes

    return result, x


@app.post("/predict/tabular")
async def predict_tabular(req: TabularRawRequest, response: Response, request: Request,
                           identity: dict = Depends(check_rate_limit)):
    result, x = await asyncio.to_thread(_score_tabular, req.features)
    # [RULE 4] the PSI / window work is NOT done here: the preprocessed row is queued for the background monitor (check_rate_limit
    # registered the BackgroundTask) and the header reflects the LAST completed evaluation -- a dict read, no computation.
    request.state.monitor_row = x
    if _drift_state.get("drifted"):
        response.headers["X-Model-Drift-Warning"] = "true"
    return result


@app.get("/health")
def health(response: Response):
    # Deliberately NOT behind require_api_key: load balancers / k8s probes must reach it without a
    # credential. It exposes only readiness flags, never predictions or model internals.
    # Returns a real 503 when this model's artifact failed to load -- or when authentication is required but no key store is
    # available (auth_ready=false) -- so k8s's readinessProbe keeps a broken pod out of the load balancer instead of marking it Ready.
    _required_loaded = (_session is not None or _linear_model is not None or _tree_ensemble is not None) and _recipe is not None and _threshold_ready()
    _auth_ok = _auth_ready()
    if not (_required_loaded and _auth_ok):
        response.status_code = 503
    return {"status": "ok" if (_required_loaded and _auth_ok) else ("auth_not_ready" if _required_loaded else "model_not_loaded"),
            "model_type": "TABULAR_ML",
            "model_loaded": _required_loaded,
            "auth_enforced": REQUIRE_AUTH,
            "auth_ready": _auth_ok,
            "airgapped": os.environ.get("AIRGAPPED_MODE", "false").lower() == "true",
            "onnx_loaded": _session is not None,
            "linear_model_loaded": _linear_model is not None,
            "tree_ensemble_loaded": _tree_ensemble is not None,
            "recipe_loaded": _recipe is not None,
            "decision_threshold_ready": _threshold_ready(),
            "drift_monitoring_active": _drift_baseline is not None,
            "artifact_load_errors": (_ARTIFACT_LOAD_ERRORS if _HEALTH_VERBOSE else sorted(_ARTIFACT_LOAD_ERRORS))}


@app.get("/healthz")
def healthz():
    # LIVENESS only: the process is up and serving HTTP. Deliberately independent of model / key-store state, so a pod that is merely
    # waiting for a Secret or still loading weights is never restarted by the livenessProbe (readiness uses /health).
    return {"status": "alive"}


@app.get("/monitoring")
def monitoring(identity: dict = Depends(require_api_key)):
    """[RULE 4] Rolling latency (p50 / p95 / max over the last DRIFT_WINDOW_SIZE requests) and, for models that ship a drift baseline, the
    latest PSI evaluation. Read-only view of what the background monitor has collected; authenticated like the prediction endpoints."""
    return _monitoring_snapshot()


# ---- Startup artifact report: one line per artifact path this bundle looks for ----------------------------
def _report_artifacts():
    for _name in sorted(k for k, v in list(globals().items()) if k.endswith("_PATH") and isinstance(v, str)):
        _p = globals()[_name]
        if _p and os.path.exists(_p):
            print("[Startup] artifact %s: found -> %s" % (_name, _p))
        elif _p:
            print("[Startup] artifact %s: MISSING -> %s (named in app.py but not in the bundle)" % (_name, _p))
        else:
            print("[Startup] artifact %s: not shipped (fine when this model uses another artifact)" % _name)


_report_artifacts()

# the helper only exists in apps that serve a tabular/ANN classifier (NLP / CNN / RNN / RAG / ... apps don't carry it)
_startup_thr = globals()["_decision_threshold_cfg"]() if "_decision_threshold_cfg" in globals() else None
if _startup_thr is not None:
    print("[Startup] manual decision threshold %s (positive_index=%d) loaded from feature_recipe.json -- binary predictions "
          "use P(positive) >= this manual threshold." % (_startup_thr["threshold"], _startup_thr["positive_index"]))
elif "_threshold_required" in globals() and globals()["_threshold_required"]():
    # [STARTUP WARNING] without a threshold every binary /predict answers HTTP 500 (the automatic 0.50 default was removed on
    # purpose). /health stays green, so the first caller used to be the one to find out -- say it in the pod log at boot.
    print("[Startup] WARNING: this is a BINARY classifier but feature_recipe.json has no decision_threshold and the "
          "DECISION_THRESHOLD env var is unset -- every prediction will answer HTTP 500 until one is set "
          "(e.g. DECISION_THRESHOLD=0.35).")
