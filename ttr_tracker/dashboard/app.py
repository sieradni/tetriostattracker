import json
import os
import tempfile
from datetime import datetime
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

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = os.environ.get("TTR_DB_PATH", str(BASE_DIR / "data" / "tracker.db"))
REPLAY_DIR = BASE_DIR / "data" / "replays"

db = Database(DB_PATH)
REPLAY_DIR.mkdir(parents=True, exist_ok=True)

static_dir = Path(__file__).resolve().parent / "static"
templates_dir = Path(__file__).resolve().parent / "templates"

app = FastAPI(title="Tetrio Stats Tracker")
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


def serve_html(name: str) -> str:
    return (templates_dir / name).read_text(encoding="utf-8")


# ── Pages ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return serve_html("index.html")


@app.get("/replays", response_class=HTMLResponse)
async def replays_page():
    return serve_html("replays.html")


@app.get("/replays/{replay_id}", response_class=HTMLResponse)
async def replay_detail_page(replay_id: str):
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
async def trends_page():
    return serve_html("trends.html")


@app.get("/sessions", response_class=HTMLResponse)
async def sessions_page():
    return serve_html("sessions.html")


@app.get("/pbs", response_class=HTMLResponse)
async def pbs_page():
    return serve_html("pbs.html")


@app.get("/partial-runs", response_class=HTMLResponse)
async def partial_runs_page():
    return serve_html("partial_runs.html")


@app.get("/ocr-test", response_class=HTMLResponse)
async def ocr_test_page():
    return serve_html("ocr_test.html")


@app.get("/blitz", response_class=HTMLResponse)
async def blitz_stats_page():
    return serve_html("blitz_stats.html")


# ── Gamemodes ─────────────────────────────────────────────────────────────

@app.get("/api/gamemodes")
async def api_gamemodes():
    with db.session() as sess:
        from sqlalchemy import func
        rows = sess.query(ReplayRow.gamemode).distinct().order_by(ReplayRow.gamemode).all()
    modes = [r[0] for r in rows]
    if "blitz" not in modes:
        modes.insert(0, "blitz")
    return modes


# ── Summary / Trends / Sessions ──────────────────────────────────────────

@app.get("/api/summary")
async def api_summary(gamemode: str = Query("blitz")):
    return get_summary(db, gamemode)


@app.get("/api/trends")
async def api_trends(gamemode: str = Query("blitz")):
    return get_trends(db, gamemode)


@app.get("/api/sessions")
async def api_sessions(gamemode: str = Query("blitz")):
    return get_session_summaries(db, gamemode)


# ── Replay Import ─────────────────────────────────────────────────────────

@app.post("/api/import")
async def api_import(files: list[UploadFile] = File(...)):
    results = []
    for f in files:
        if not f.filename or not f.filename.endswith(".ttr"):
            results.append({"file": f.filename or "unknown", "status": "skipped", "message": "not a .ttr file"})
            continue
        try:
            dest = REPLAY_DIR / f.filename
            content = await f.read()
            dest.write_bytes(content)
            msg = import_file(db, dest)
            results.append({"file": f.filename, "status": "ok", "message": msg})
        except Exception as e:
            results.append({"file": f.filename, "status": "error", "message": str(e)})
    return results


# ── Combined Replays + Partial Runs API ──────────────────────────────────

@app.get("/api/replays")
async def api_replays(
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
async def api_replay_detail(replay_id: str):
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

def _compute_partial_fields(score: int, pieces: int, pps: float, inputs: int,
                             kpp: float, spp: float, all_clears: int,
                             time_left: float) -> dict:
    objective_ms = 120000
    time_left_ms = time_left * 1000
    time_elapsed = max(0, objective_ms - time_left_ms) / 1000
    kps = inputs / time_elapsed if time_elapsed > 0 else 0.0
    return {
        "time_elapsed": time_elapsed,
        "kps": round(kps, 4),
    }


@app.post("/api/partial-runs")
async def api_create_partial_run(
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
        ts = datetime.fromisoformat(timestamp)
    except Exception:
        return JSONResponse({"error": "invalid timestamp"}, status_code=400)

    computed = _compute_partial_fields(score, pieces, pps, inputs, kpp, spp, all_clears, time_left)
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
        ts = datetime.fromisoformat(timestamp)
    except Exception:
        return JSONResponse({"error": "invalid timestamp"}, status_code=400)

    computed = _compute_partial_fields(score, pieces, pps, inputs, kpp, spp, all_clears, time_left)
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

    test_images_dir = BASE_DIR / "test_images"
    test_images_dir.mkdir(parents=True, exist_ok=True)
    image_ext = Path(image.filename or "screenshot.png").suffix
    image_filename = f"ocr_test_{rid}_{score}_{pieces}{image_ext}"
    image_path = test_images_dir / image_filename
    content = await image.read()
    image_path.write_bytes(content)

    db.update_partial_run(rid, source_image=str(image_path))

    return {"id": rid, "status": "ok", "image_filename": image_filename}


@app.get("/api/partial-runs")
async def api_list_partial_runs(
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
async def api_get_partial_run(run_id: int):
    detail = get_partial_run_detail(db, run_id)
    if not detail:
        return JSONResponse({"error": "not found"}, status_code=404)
    return detail


@app.delete("/api/partial-runs/{run_id}")
async def api_delete_partial_run(run_id: int):
    ok = db.delete_partial_run(run_id)
    if not ok:
        return JSONResponse({"error": "not found"}, status_code=404)
    return {"status": "deleted"}


# ── Blitz Stats / Survival ───────────────────────────────────────────────

@app.get("/api/blitz/survival")
async def api_blitz_survival(gamemode: str = Query("blitz")):
    return get_blitz_survival(db, gamemode)


# ── OCR ──────────────────────────────────────────────────────────────────

@app.get("/api/ocr/status")
async def api_ocr_status():
    return {"available": is_available()}


@app.post("/api/ocr/extract")
async def api_ocr_extract(file: UploadFile = File(...)):
    if not is_available():
        return JSONResponse(
            {"error": "Tesseract OCR is not installed. Install from https://github.com/UB-Mannheim/tesseract/wiki"},
            status_code=422,
        )
    suffix = Path(file.filename or "screenshot.png").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        content = await file.read()
        f.write(content)
        tmp = f.name
    try:
        result = extract_stats(tmp)
        return result
    except TesseractNotAvailableError as e:
        return JSONResponse({"error": str(e)}, status_code=422)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=422)
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass


# ── Match / League ───────────────────────────────────────────────────────

@app.get("/matches", response_class=HTMLResponse)
async def matches_page():
    return serve_html("matches.html")


@app.get("/matches/{match_id}", response_class=HTMLResponse)
async def match_detail_page(match_id: str):
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
async def api_matches(limit: int = Query(100), offset: int = Query(0)):
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
async def api_match_detail(match_id: str):
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
async def api_match_trends(player_id: str, limit: int = Query(200)):
    data = db.get_match_aggregate_trends(player_id, limit)
    return data


@app.get("/api/matches/leaderboard/{player_id}")
async def api_match_leaderboard(player_id: str):
    from ttr_tracker.queries import get_match_leaderboard
    return get_match_leaderboard(db, player_id)


@app.get("/api/matches/per_round_trends")
async def api_match_per_round_trends(player_name: str = Query(...)):
    from ttr_tracker.queries import get_match_trends
    return get_match_trends(db, player_name)
