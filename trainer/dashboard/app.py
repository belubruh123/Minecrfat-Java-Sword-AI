"""Training dashboard: replays recorded arena-0 episodes (the agent's-eye
view) at real-time speed and charts training metrics.

Usage: uvicorn dashboard.app:app --host 0.0.0.0 --port 8080
(from trainer/, with the repo venv active)
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse

RUNS_DIR = Path(__file__).resolve().parent.parent.parent / "runs"
STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="DRL Agent Dashboard")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/runs")
def runs():
    out = []
    if RUNS_DIR.is_dir():
        for d in RUNS_DIR.iterdir():
            if not d.is_dir():
                continue
            metrics = d / "metrics.jsonl"
            eps = list((d / "episodes").glob("ep_*.npz")) if (d / "episodes").is_dir() else []
            if not metrics.exists() and not eps:
                continue
            out.append({
                "name": d.name,
                "episodes": len(eps),
                "updated": max([metrics.stat().st_mtime if metrics.exists() else 0]
                               + [e.stat().st_mtime for e in eps]),
            })
    out.sort(key=lambda r: -r["updated"])
    return out


def _run_dir(run: str) -> Path:
    d = (RUNS_DIR / run).resolve()
    if not d.is_dir() or d.parent != RUNS_DIR.resolve():
        raise HTTPException(404, "no such run")
    return d


@app.get("/api/run/{run}/metrics")
def metrics(run: str, tail: int = 800):
    f = _run_dir(run) / "metrics.jsonl"
    if not f.exists():
        return []
    lines = f.read_text().strip().splitlines()[-tail:]
    return JSONResponse([json.loads(x) for x in lines if x])


@app.get("/api/run/{run}/episodes")
def episodes(run: str):
    d = _run_dir(run) / "episodes"
    out = []
    if d.is_dir():
        for meta in d.glob("ep_*.json"):
            try:
                m = json.loads(meta.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if (d / f"{m.get('name', '')}.npz").exists():
                out.append(m)
    out.sort(key=lambda m: -m.get("ts", 0))
    return out[:30]


@app.get("/api/run/{run}/episode/{name}")
def episode(run: str, name: str, light: int = 0):
    if not name.replace("_", "").isalnum():
        raise HTTPException(400, "bad name")
    f = _run_dir(run) / "episodes" / f"{name}.npz"
    if not f.exists():
        raise HTTPException(404, "no such episode")
    z = np.load(f)
    out = {
        "shape": z["shape"].tolist(),
        "fps": 20,
        "actions": z["actions"].tolist(),
        "rewards": z["rewards"].tolist(),
        "infos": z["infos"].tolist(),
    }
    if not light:  # the replay theater only needs telemetry + infos
        out["frames"] = [base64.b64encode(row.tobytes()).decode() for row in z["masks"]]
        out["scalars"] = np.round(z["scalars"].astype(float), 3).tolist()
    if "telemetry" in z:  # older recordings predate the top-down view
        out["telemetry"] = np.round(z["telemetry"].astype(float), 3).tolist()
    return JSONResponse(out)


LIVE_JPG = RUNS_DIR / "live.jpg"


@app.get("/api/live")
def live():
    """Latest frame of the replay-theater client (ffmpeg keeps overwriting
    runs/live.jpg). 404 when the theater is not running or the frame is
    stale, so the page can hide the panel."""
    import time
    if not LIVE_JPG.exists() or time.time() - LIVE_JPG.stat().st_mtime > 5:
        raise HTTPException(404, "no live frame")
    return FileResponse(LIVE_JPG, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})
