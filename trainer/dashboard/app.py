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


LADDER = RUNS_DIR / "ladder.json"


@app.get("/api/progress")
def progress():
    """Training pipeline with real-time ETA math. runs/ladder.json names the
    stages (maintained by the training orchestration); live step counts,
    sps and headline stats come from each run's metrics.jsonl."""
    import time
    if not LADDER.exists():
        raise HTTPException(404, "no ladder")
    ladder = json.loads(LADDER.read_text())
    now = time.time()
    ref_sps = 200.0  # queued-stage estimate until a live rate is known
    stages = []
    for st in ladder.get("stages", []):
        row = dict(st)
        f = (RUNS_DIR / st["run"] / "metrics.jsonl") if st.get("run") else None
        if f is not None and f.exists():
            recs = [json.loads(x) for x in
                    f.read_text().strip().splitlines()[-12:] if x]
            recs = [r for r in recs if "step" in r]
            if recs:
                last = recs[-1]
                sps = float(np.mean([r["sps"] for r in recs if r.get("sps")]))
                row["step"] = last["step"]
                row["sps"] = round(sps, 1)
                row["age_s"] = round(now - f.stat().st_mtime)
                row["stats"] = {k: round(float(v), 2) for k, v in last.items()
                                if k in ("success", "ep_hits", "ep_whiffs",
                                         "ep_chain_hits", "ep_hits_taken",
                                         "ep_return")}
                if last["step"] >= st.get("total_steps", 1):
                    row["status"] = "done"
                elif row.get("status") == "running":
                    row["stalled"] = row["age_s"] > 180
                    if sps > 0:
                        ref_sps = sps
                        row["eta_s"] = (st["total_steps"] - last["step"]) / sps
        stages.append(row)
    total = 0.0
    for row in stages:
        if row.get("status") == "queued" and "eta_s" not in row:
            row["eta_s"] = (row["fixed_minutes"] * 60 if row.get("fixed_minutes")
                            else row.get("total_steps", 0) / ref_sps)
        if row.get("status") in ("running", "queued"):
            total += row.get("eta_s", 0)
    return {"now": now, "total_eta_s": round(total), "stages": stages}


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
