"""
server.py
=========
FastAPI backend for the Market Stage Dashboard.

Endpoints
---------
  GET  /                   → serves dashboard.html
  GET  /api/status         → { status, last_updated, progress }
  POST /api/analyze        → triggers background analysis
  GET  /api/report         → returns cached JSON report
  GET  /api/tickers        → flat ticker list with all metrics

Run
---
  pip install fastapi uvicorn yfinance pandas numpy tabulate
  python server.py

Then open http://localhost:8000
"""

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, List

import numpy as np
import pandas as pd
from fastapi.encoders import jsonable_encoder

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# ── Import your analyzer modules (must be in same directory) ──────────────────
from market_stage_analyzer import MarketAnalyzer

ROOT_DIR = Path(__file__).resolve().parent.parent   # repo root
DASHBOARD_PATH = ROOT_DIR / "dashboard.html"

# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Market Stage Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ──────────────────────────────────────────────────────────────
_state: Dict[str, Any] = {
    "status": "idle",  # idle | running | ready | error
    "last_updated": None,
    "progress": "",
    "report": None,  # serialised report dict
    "error": None,
}
_lock = threading.Lock()

CACHE_FILE = Path("market_cache.json")
CACHE_TTL = 3600  # seconds — re-use cached data for 1 hour


# ──────────────────────────────────────────────────────────────────────────────
# Serialisation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _to_builtin(x: Any) -> Any:
    """Convert numpy/pandas scalars to plain Python types for JSON encoding."""
    if x is None:
        return None
    # numpy scalar (np.float64, np.int64, np.bool_, etc.)
    if isinstance(x, np.generic):
        return x.item()
    # pandas Timestamp / Timedelta
    if isinstance(x, (pd.Timestamp, pd.Timedelta)):
        return x.isoformat()
    return x


def _json_default(o: Any) -> Any:
    """json.dumps default handler for non-serializable objects."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, (pd.Timestamp, pd.Timedelta)):
        return o.isoformat()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _ticker_to_dict(t) -> dict:
    return {
        "symbol": t.symbol,
        "sector": t.sector,
        "stage": int(_to_builtin(t.stage)),
        "stage_label": t.stage_label,
        "price": round(float(_to_builtin(t.price)), 2),
        "ma_50": round(float(_to_builtin(t.ma_50)), 2),
        "ma_150": round(float(_to_builtin(t.ma_150)), 2),
        "ma_150_slope": round(float(_to_builtin(t.ma_150_slope)), 2),
        "rsi": round(float(_to_builtin(t.rsi)), 1),
        "perf_1w": round(float(_to_builtin(t.perf_1w)), 2),
        "perf_1m": round(float(_to_builtin(t.perf_1m)), 2),
        "perf_3m": round(float(_to_builtin(t.perf_3m)), 2),
        "price_vs_ma150": round(float(_to_builtin(t.price_vs_ma150)), 2),
        "pct_from_52w_high": round(float(_to_builtin(t.pct_from_52w_high)), 2),
        "is_up": bool(_to_builtin(t.is_up)),
        "error": t.error,
    }


def _sector_to_dict(s) -> dict:
    return {
        "sector": s.sector,
        "stage": int(_to_builtin(s.stage)),
        "stage_label": s.stage_label,
        "trend": s.trend,
        "avg_rsi": round(float(_to_builtin(s.avg_rsi)), 1),
        "avg_perf_1w": round(float(_to_builtin(s.avg_perf_1w)), 2),
        "avg_perf_1m": round(float(_to_builtin(s.avg_perf_1m)), 2),
        "avg_perf_3m": round(float(_to_builtin(s.avg_perf_3m)), 2),
        "pct_stage2": round(float(_to_builtin(s.pct_stage2)), 1),
        "pct_stage4": round(float(_to_builtin(s.pct_stage4)), 1),
        "ticker_count": len(s.tickers),
        "tickers": [_ticker_to_dict(t) for t in s.tickers],
    }


def _report_to_dict(report) -> dict:
    return {
        "timestamp": report.timestamp,
        "overall_trend": report.overall_trend,
        "bull_sectors": report.bull_sectors,
        "bear_sectors": report.bear_sectors,
        "sectors": {k: _sector_to_dict(v) for k, v in report.sectors.items()},
    }


# ──────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ──────────────────────────────────────────────────────────────────────────────

def _save_cache(data: dict):
    # Ensure all numpy/pandas scalars are converted to JSON-friendly builtins.
    payload = jsonable_encoder(data)
    payload["_cached_at"] = time.time()
    CACHE_FILE.write_text(json.dumps(payload, indent=2, default=_json_default))

def _load_cache() -> Optional[dict]:
    if not CACHE_FILE.exists():
        return None
    try:
        data = json.loads(CACHE_FILE.read_text())
        age = time.time() - data.get("_cached_at", 0)
        if age < CACHE_TTL:
            return data
    except Exception:
        pass
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Background analysis worker
# ──────────────────────────────────────────────────────────────────────────────

def _run_analysis(tickers: List[str]):
    """Runs in a background thread."""
    try:
        with _lock:
            _state["status"] = "running"
            _state["progress"] = "Downloading price data…"
            _state["error"] = None

        analyzer = MarketAnalyzer(tickers)
        report = analyzer.run()
        data = _report_to_dict(report)

        _save_cache(data)

        with _lock:
            _state["status"] = "ready"
            _state["report"] = data
            _state["last_updated"] = data["timestamp"]
            _state["progress"] = ""

    except Exception as exc:
        with _lock:
            _state["status"] = "error"
            _state["error"] = str(exc)
            _state["progress"] = ""


# ──────────────────────────────────────────────────────────────────────────────
# Startup: try loading cache
# ──────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
def _startup():
    cached = _load_cache()
    if cached:
        with _lock:
            _state["status"] = "ready"
            _state["report"] = cached
            _state["last_updated"] = cached.get("timestamp")
        print("[server] Loaded cached report from disk.")
    else:
        print("[server] No valid cache found. Use POST /api/analyze to fetch data.")


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def serve_dashboard():
    path = Path(DASHBOARD_PATH)
    if not path.exists():
        raise HTTPException(404, "dashboard.html not found in working directory")
    return FileResponse(path)


@app.get("/api/status")
def get_status():
    with _lock:
        return {
            "status": _state["status"],
            "last_updated": _state["last_updated"],
            "progress": _state["progress"],
            "error": _state["error"],
        }


@app.post("/api/analyze")
def trigger_analysis():
    with _lock:
        if _state["status"] == "running":
            return {"message": "Analysis already running"}

    with open("../res/can_tickers", "r") as f:
        tickers: List[str] = [line.strip() for line in f if line.strip()]

    t = threading.Thread(
        target=_run_analysis,
        args=(tickers, ),
        daemon=True,
    )
    t.start()
    return {"message": "Analysis started"}


@app.get("/api/report")
def get_report():
    with _lock:
        if _state["status"] == "idle":
            raise HTTPException(404, "No report yet. POST /api/analyze to start.")
        if _state["status"] == "running":
            raise HTTPException(202, "Analysis in progress")
        if _state["status"] == "error":
            raise HTTPException(500, _state["error"])
        return _state["report"]


@app.get("/api/tickers")
def get_tickers():
    """Flat list of all ticker results across all sectors."""
    with _lock:
        report = _state.get("report")
    if not report:
        raise HTTPException(404, "No report available")

    rows = []
    for sec_data in report["sectors"].values():
        for t in sec_data["tickers"]:
            rows.append({**t, "sector_trend": sec_data["trend"]})
    return rows


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
