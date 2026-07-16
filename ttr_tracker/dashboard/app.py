import asyncio
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ttr_tracker.database import Database, ReplayRow, MatchRow
from ttr_tracker.importer import import_file
from ttr_tracker.ocr import extract_stats, is_available, TesseractNotAvailableError
from ttr_tracker.queries import (
    get_blitz_survival,
    get_partial_run_detail,
    get_replay_detail,
    get_session_summaries,
    get_summary,
    get_trends,
    get_match_leaderboard,
    get_match_trends,
)
from ttr_tracker.stats import compute_kps, compute_time_elapsed
from sqlalchemy import func

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = os.environ.get("TTR_DB_PATH", str(BASE_DIR / "data" / "tracker.db"))
REPLAY_DIR = BASE_DIR / "data" / "replays"

MAX_TTR_FILE_SIZE = 5 * 1024 * 1024       # 5 MB
MAX_OCR_IMAGE_SIZE = 20 * 1024 * 1024      # 20 MB

db = Database(DB_PATH)
REPLAY_DIR.mkdir(parents=True, exist_ok=True)

static_dir = Path(__file__).resolve().parent / "static"
templates_dir = Path(__file__).resolve().parent / "templates"

app = FastAPI(title="Tetrio Stats Tracker")
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


def serve_html(name: str) -> str:
    full_path = (templates_dir / name).resolve()
    if not str(full_path).startswith(str(templates_dir.resolve())):
        raise ValueError("Invalid template name")
    return full_path.read_text(encoding="utf-8")


# ── Pages ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    return serve_html("index.html")


@app.get("/replays", response_class=HTMLResponse)
def replays_page():
    return serve_html("replays.html")


@app.get("/replays/{replay_id}", response_class=HTMLResponse)
def replay_detail_page(replay_id: str):
    if replay_id.startswith("partial-"):
        pid = int(replay_id.removeprefix("partial-"))
        detail = get_partial_run_detail(db, pid)
        if not detail:
            return HTMLResponse("Partial run not found", status_code=404)
        html = serve_html("partial_run_detail.html")
        html = html.replace("/*{{PARTIAL_DATA}}*/", json.dumps(detail))
        return html

    detail = get_replay_detail(db, replay_id)
    if not detail:
        return HTMLResponse("Replay not found", status_code=404)
    html = serve_html("replay_detail.html")
    html = html.replace("/*{{REPLAY_DATA}}*/", json.dumps(detail))
    return html


@app.get("/trends", response_class=HTMLResponse)
def trends_page():
    return serve_html("trends.html")


@app.get("/sessions", response_class=HTMLResponse)
def sessions_page():
    return serve_html("sessions.html")


@app.get("/pbs", response_class=HTMLResponse)
def pbs_page():
    return serve_html("pbs.html")


@app.get("/partial-runs", response_class=HTMLResponse)
def partial_runs_page():
    return serve_html("partial_runs.html")


@app.get("/ocr-test", response_class=HTMLResponse)
def ocr_test_page():
    return serve_html("ocr_test.html")


@app.get("/blitz", response_class=HTMLResponse)
def blitz_stats_page():
    return serve_html("blitz_stats.html")


# ── Gamemodes ─────────────────────────────────────────────────────────────

@app.get("/api/gamemodes")
def api_gamemodes():
    with db.session() as sess:
        rows = sess.query(ReplayRow.gamemode).distinct().order_by(ReplayRow.gamemode).all()
    modes = [r[0] for r in rows]
    if "blitz" not in modes:
        modes.insert(0, "blitz")
    return modes


# ── Summary / Trends / Sessions ──────────────────────────────────────────

@app.get("/api/summary")
def api_summary(gamemode: str = Query("blitz")):
    return get_summary(db, gamemode)


@app.get("/api/trends")
def api_trends(gamemode: str = Query("blitz")):
    return get_trends(db, gamemode)


@app.get("/api/sessions")
def api_sessions(gamemode: str = Query("blitz")):
    return get_session_summaries(db, gamemode)


# ── Replay Import ─────────────────────────────────────────────────────────

@app.post("/api/import")
async def api_import(files: list[UploadFile] = File(...)):
    # Read file contents async first, then process (sync) in thread pool
    file_tasks = []
    for f in files:
        safe_name = Path(f.filename or "unknown").name
        if not f.filename or not (f.filename.endswith(".ttr") or f.filename.endswith(".ttrm")):
            file_tasks.append(("skip", safe_name, b"", "not a .ttr or .ttrm file"))
            continue
        content = await f.read()
        if len(content) > MAX_TTR_FILE_SIZE:
            file_tasks.append(("skip", safe_name, b"", f"file too large ({len(content) / 1024 / 1024:.1f} MB)"))
            continue
        dest = REPLAY_DIR / safe_name
        file_tasks.append(("import", safe_name, content, dest))

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: _process_imports(db, file_tasks))


def _process_imports(db: Database, file_tasks: list) -> list[dict]:
    results = []
    for task in file_tasks:
        if task[0] == "skip":
            results.append({"file": task[1], "status": "skipped", "message": task[3] if len(task) > 3 else "not a .ttr file"})
        elif task[0] == "import":
            safe_name, content, dest = task[1], task[2], task[3]
            try:
                dest.write_bytes(content)
                msg = import_file(db, dest)
                results.append({"file": safe_name, "status": "ok", "message": msg})
            except Exception as e:
                results.append({"file": safe_name, "status": "error", "message": str(e)})
    return results


# ── Combined Replays + Partial Runs API ──────────────────────────────────

@app.get("/api/replays")
def api_replays(
    gamemode: Optional[str] = Query(None),
    limit: int = Query(200),
    offset: int = Query(0),
):
    entries = db.get_combined_entries(gamemode=gamemode, limit=limit + offset)
    entries = entries[offset:]
    for e in entries:
        e["timestamp"] = e["timestamp"].isoformat() if hasattr(e["timestamp"], "isoformat") else e["timestamp"]
    return entries


@app.get("/api/replays/{replay_id}")
def api_replay_detail(replay_id: str):
    if replay_id.startswith("partial-"):
        pid = int(replay_id.removeprefix("partial-"))
        detail = get_partial_run_detail(db, pid)
        if not detail:
            return JSONResponse({"error": "not found"}, status_code=404)
        return detail
    detail = get_replay_detail(db, replay_id)
    if not detail:
        return JSONResponse({"error": "not found"}, status_code=404)
    return detail


# ── Partial Runs CRUD ────────────────────────────────────────────────────

def _compute_partial_fields(inputs: int, time_left: float) -> dict:
    time_elapsed = compute_time_elapsed(time_left)
    kps = compute_kps(inputs, time_elapsed)
    return {
        "time_elapsed": time_elapsed,
        "kps": round(kps, 4),
    }


@app.post("/api/partial-runs")
def api_create_partial_run(
    timestamp: str = Form(...),
    score: int = Form(...),
    pieces: int = Form(...),
    pps: float = Form(...),
    inputs: int = Form(...),
    kpp: float = Form(...),
    spp: float = Form(...),
    all_clears: int = Form(...),
    time_left: float = Form(...),
    notes: Optional[str] = Form(None),
):
    try:
        ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except Exception:
        return JSONResponse({"error": "invalid timestamp"}, status_code=400)

    computed = _compute_partial_fields(inputs, time_left)
    rid = db.insert_partial_run(
        timestamp=ts,
        gamemode="blitz",
        score=score,
        pieces_placed=pieces,
        pps=pps,
        inputs=inputs,
        kpp=kpp,
        spp=spp,
        all_clears=all_clears,
        time_left=time_left,
        time_elapsed=computed["time_elapsed"],
        kps=computed["kps"],
        notes=notes,
    )
    return {"id": rid, "status": "ok"}


@app.post("/api/ocr-test/save")
async def api_ocr_test_save(
    image: UploadFile = File(...),
    timestamp: str = Form(...),
    score: int = Form(...),
    pieces: int = Form(...),
    pps: float = Form(...),
    inputs: int = Form(...),
    kpp: float = Form(...),
    spp: float = Form(...),
    all_clears: int = Form(...),
    time_left: float = Form(...),
    notes: Optional[str] = Form(None),
):
    try:
        ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except Exception:
        return JSONResponse({"error": "invalid timestamp"}, status_code=400)

    image_data = await image.read()
    if len(image_data) > MAX_OCR_IMAGE_SIZE:
        return JSONResponse({"error": f"Image too large ({len(image_data) / 1024 / 1024:.1f} MB, max {MAX_OCR_IMAGE_SIZE / 1024 / 1024:.0f} MB)"}, status_code=413)

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: _save_ocr_test(db, image_data, image.filename, ts, score, pieces, pps, inputs, kpp, spp, all_clears, time_left, notes))


def _save_ocr_test(db: Database, image_data: bytes, image_filename: str | None, ts: datetime, score: int, pieces: int, pps: float, inputs: int, kpp: float, spp: float, all_clears: int, time_left: float, notes: Optional[str]) -> dict:
    computed = _compute_partial_fields(inputs, time_left)
    rid = db.insert_partial_run(
        timestamp=ts,
        gamemode="blitz",
        score=score,
        pieces_placed=pieces,
        pps=pps,
        inputs=inputs,
        kpp=kpp,
        spp=spp,
        all_clears=all_clears,
        time_left=time_left,
        time_elapsed=computed["time_elapsed"],
        kps=computed["kps"],
        notes=notes,
    )

    ocr_images_dir = BASE_DIR / "data" / "ocr_test_images"
    ocr_images_dir.mkdir(parents=True, exist_ok=True)
    image_ext = Path(image_filename or "screenshot.png").suffix
    image_name = f"ocr_test_{rid}_{score}_{pieces}{image_ext}"
    image_path = ocr_images_dir / image_name
    image_path.write_bytes(image_data)

    db.update_partial_run(rid, source_image=str(image_path))

    return {"id": rid, "status": "ok", "image_filename": image_name}


@app.get("/api/partial-runs")
def api_list_partial_runs(
    gamemode: Optional[str] = Query(None),
    limit: int = Query(100),
    offset: int = Query(0),
):
    rows = db.get_all_partial_runs(gamemode=gamemode, limit=limit, offset=offset)
    return [
        {
            "id": r.id,
            "timestamp": r.timestamp.isoformat(),
            "gamemode": r.gamemode,
            "score": r.score,
            "pieces_placed": r.pieces_placed,
            "pps": r.pps,
            "kpp": r.kpp,
            "kps": r.kps,
            "spp": r.spp,
            "all_clears": r.all_clears,
            "time_left": r.time_left,
            "time_elapsed": r.time_elapsed,
            "notes": r.notes,
        }
        for r in rows
    ]


@app.get("/api/partial-runs/{run_id}")
def api_get_partial_run(run_id: int):
    detail = get_partial_run_detail(db, run_id)
    if not detail:
        return JSONResponse({"error": "not found"}, status_code=404)
    return detail


@app.put("/api/partial-runs/{run_id}")
def api_update_partial_run(
    run_id: int,
    timestamp: str = Form(...),
    score: int = Form(...),
    pieces: int = Form(...),
    pps: float = Form(...),
    inputs: int = Form(...),
    kpp: float = Form(...),
    spp: float = Form(...),
    all_clears: int = Form(...),
    time_left: float = Form(...),
    notes: Optional[str] = Form(None),
):
    try:
        ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except Exception:
        return JSONResponse({"error": "invalid timestamp"}, status_code=400)

    computed = _compute_partial_fields(inputs, time_left)
    ok = db.update_partial_run(
        run_id,
        timestamp=ts,
        score=score,
        pieces_placed=pieces,
        pps=pps,
        inputs=inputs,
        kpp=kpp,
        spp=spp,
        all_clears=all_clears,
        time_left=time_left,
        time_elapsed=computed["time_elapsed"],
        kps=computed["kps"],
        notes=notes,
    )
    if not ok:
        return JSONResponse({"error": "not found"}, status_code=404)
    return {"id": run_id, "status": "ok"}


@app.delete("/api/partial-runs/{run_id}")
def api_delete_partial_run(run_id: int):
    run = db.get_partial_run(run_id)
    if not run:
        return JSONResponse({"error": "not found"}, status_code=404)
    db.delete_partial_run(run_id)
    if run.source_image and os.path.isfile(run.source_image):
        try:
            os.remove(run.source_image)
        except OSError as e:
            print(f"Warning: could not delete source image {run.source_image}: {e}")
    return {"status": "deleted"}


@app.delete("/api/replays/{replay_id}")
def api_delete_replay(replay_id: str):
    ok = db.delete_replay(replay_id)
    if not ok:
        return JSONResponse({"error": "not found"}, status_code=404)
    return {"status": "deleted"}


@app.get("/partial-runs/{run_id}/edit", response_class=HTMLResponse)
def partial_run_edit_page(run_id: int):
    detail = get_partial_run_detail(db, run_id)
    if not detail:
        return HTMLResponse("Partial run not found", status_code=404)
    html = serve_html("partial_run_edit.html")
    html = html.replace("/*{{RUN_ID}}*/", str(run_id))
    return html


# ── Blitz Stats / Survival ───────────────────────────────────────────────

@app.get("/api/blitz/survival")
def api_blitz_survival(gamemode: str = Query("blitz"),
                       start_date: Optional[str] = Query(None),
                       end_date: Optional[str] = Query(None)):
    start = datetime.fromisoformat(start_date) if start_date else None
    end = datetime.fromisoformat(end_date) if end_date else None
    return get_blitz_survival(db, gamemode, start, end)


# ── OCR ──────────────────────────────────────────────────────────────────

@app.get("/api/ocr/status")
def api_ocr_status():
    return {"available": is_available()}


@app.post("/api/ocr/extract")
async def api_ocr_extract(file: UploadFile = File(...)):
    if not is_available():
        return JSONResponse(
            {"error": "Tesseract OCR is not installed. Install from https://github.com/UB-Mannheim/tesseract/wiki"},
            status_code=422,
        )
    suffix = Path(file.filename or "screenshot.png").suffix
    content = await file.read()
    if len(content) > MAX_OCR_IMAGE_SIZE:
        return JSONResponse({"error": f"Image too large ({len(content) / 1024 / 1024:.1f} MB, max {MAX_OCR_IMAGE_SIZE / 1024 / 1024:.0f} MB)"}, status_code=413)

    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(None, lambda: _run_ocr(content, suffix))
    except TesseractNotAvailableError as e:
        return JSONResponse({"error": str(e)}, status_code=422)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=422)


def _run_ocr(content: bytes, suffix: str) -> dict:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(content)
        tmp = f.name
    try:
        return extract_stats(tmp)
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass


# ── Match / League ───────────────────────────────────────────────────────

@app.get("/matches", response_class=HTMLResponse)
def matches_page():
    return serve_html("matches.html")


@app.get("/matches/{match_id}", response_class=HTMLResponse)
def match_detail_page(match_id: str):
    detail = db.get_match_detail(match_id)
    if not detail:
        return HTMLResponse("Match not found", status_code=404)
    html = serve_html("match_detail.html")
    match_row, rounds = detail
    data = {
        "id": match_row.id,
        "timestamp": match_row.timestamp.isoformat(),
        "player1_name": match_row.player1_name,
        "player2_name": match_row.player2_name,
        "player1_wins": match_row.player1_wins,
        "player2_wins": match_row.player2_wins,
        "winner_name": match_row.winner_name,
        "rounds": [
            {
                "round_number": r.round_number,
                "player_id": r.player_id,
                "player_name": r.player_name,
                "alive": bool(r.alive),
                "lifetime": r.lifetime,
                "apm": r.apm,
                "pps": r.pps,
                "vsscore": r.vsscore,
                "garbagesent": r.garbagesent,
                "garbagereceived": r.garbagereceived,
                "kills": r.kills,
                "btb": r.btb,
                "revives": r.revives,
            }
            for r in rounds
        ],
    }
    html = html.replace("/*{{MATCH_DATA}}*/", json.dumps(data))
    return html


@app.get("/api/matches")
def api_matches(limit: int = Query(100), offset: int = Query(0)):
    rows = db.get_all_matches(limit=limit, offset=offset)
    return [
        {
            "id": m.id,
            "timestamp": m.timestamp.isoformat(),
            "player1_name": m.player1_name,
            "player2_name": m.player2_name,
            "player1_wins": m.player1_wins,
            "player2_wins": m.player2_wins,
            "winner_name": m.winner_name,
        }
        for m in rows
    ]


@app.get("/api/matches/{match_id}")
def api_match_detail(match_id: str):
    detail = db.get_match_detail(match_id)
    if not detail:
        return JSONResponse({"error": "not found"}, status_code=404)
    match_row, rounds = detail
    return {
        "id": match_row.id,
        "timestamp": match_row.timestamp.isoformat(),
        "player1_name": match_row.player1_name,
        "player2_name": match_row.player2_name,
        "player1_wins": match_row.player1_wins,
        "player2_wins": match_row.player2_wins,
        "winner_name": match_row.winner_name,
        "rounds": [
            {
                "round_number": r.round_number,
                "player_id": r.player_id,
                "player_name": r.player_name,
                "alive": bool(r.alive),
                "lifetime": r.lifetime,
                "apm": r.apm,
                "pps": r.pps,
                "vsscore": r.vsscore,
                "garbagesent": r.garbagesent,
                "garbagereceived": r.garbagereceived,
                "kills": r.kills,
                "btb": r.btb,
                "revives": r.revives,
            }
            for r in rounds
        ],
    }


@app.get("/api/matches/trends/{player_id}")
def api_match_trends(player_id: str, limit: int = Query(200)):
    data = db.get_match_aggregate_trends(player_id, limit)
    return data


@app.get("/api/matches/leaderboard/{player_id}")
def api_match_leaderboard(player_id: str):
    return get_match_leaderboard(db, player_id)


@app.get("/api/matches/per_round_trends")
def api_match_per_round_trends(player_name: str = Query(...)):
    return get_match_trends(db, player_name)
