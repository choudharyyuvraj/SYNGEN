"""SYNGEN web API.

Run from the repo root:
    python server/app.py            # http://localhost:8000  (serves the built UI too)

Endpoints
    GET  /api/health          model status, device, supported characters
    POST /api/analyze         style reference -> processed style + Laplace map + metrics
    POST /api/generate        style reference + text -> job id
    GET  /api/jobs/{job_id}   job progress / results
    GET  /api/sample          the sample reference image (test.jpg)
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from engine import REPO_ROOT, Engine, parse_text  # noqa: E402
from preprocess import process_style, to_png_b64  # noqa: E402

MAX_UPLOAD = 15 * 1024 * 1024

app = FastAPI(title="SYNGEN API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
engine = Engine()


async def _read_upload(file: UploadFile) -> bytes:
    data = await file.read()
    if not data:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Image is larger than 15 MB.")
    return data


def _style(data: bytes, enhance: bool, fit: str):
    if fit not in ("aspect", "stretch"):
        raise HTTPException(400, "fit must be 'aspect' or 'stretch'.")
    try:
        return process_style(data, enhance=enhance, fit=fit)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@app.get("/api/health")
def health():
    return {"ok": True, **engine.status()}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), enhance: bool = Form(True), fit: str = Form("aspect")):
    res = _style(await _read_upload(file), enhance, fit)
    return {
        "original": to_png_b64(res.original),
        "cleaned": to_png_b64(res.cleaned),
        "style": to_png_b64(res.style),
        "laplace": to_png_b64(res.laplace),
        "metrics": res.metrics,
    }


@app.post("/api/generate")
async def generate(
    file: UploadFile = File(...),
    text: str = Form(...),
    method: str = Form("ddim"),
    steps: int = Form(50),
    eta: float = Form(0.0),
    seed: int = Form(-1),
    enhance: bool = Form(True),
    fit: str = Form("aspect"),
):
    if method not in ("ddim", "ddpm"):
        raise HTTPException(400, "method must be 'ddim' or 'ddpm'.")
    if not 5 <= steps <= 200:
        raise HTTPException(400, "steps must be between 5 and 200.")
    if not 0.0 <= eta <= 1.0:
        raise HTTPException(400, "eta must be between 0 and 1.")
    try:
        lines = parse_text(text)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    res = _style(await _read_upload(file), enhance, fit)
    if seed < 0:
        seed = random.randint(0, 2**31 - 1)
    job_id = engine.submit(res.style, res.laplace, res.metrics, lines, method, steps, eta, seed)
    return {"job_id": job_id, "seed": seed, "mode": engine.mode}


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    j = engine.get_job(job_id)
    if j is None:
        raise HTTPException(404, "Job not found (the server may have restarted).")
    return j


@app.get("/api/sample")
def sample():
    return FileResponse(REPO_ROOT / "test.jpg", media_type="image/jpeg")


# serve the production build of the frontend if it exists
DIST = REPO_ROOT / "frontend" / "dist"
if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="ui")

    @app.exception_handler(404)
    async def spa_fallback(request, exc):  # client-side routes -> index.html
        if request.url.path.startswith("/api/"):
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": getattr(exc, "detail", "Not found")}, status_code=404)
        return FileResponse(DIST / "index.html")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    print(f"\n  SYNGEN running at  http://localhost:{port}\n", flush=True)
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=port, log_level="warning")
