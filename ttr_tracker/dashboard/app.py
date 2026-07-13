import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ttr_tracker.database import Database, ReplayRow, MatchRow
from ttr_tracker.importer import import_file
from ttr_tracker.queries import (
    get_replay_detail,
    get_session_summaries,
    get_summary,
    get_trends,
    get_match_leaderboard,
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


@app.get("/", response_class=HTMLResponse)
async def index():
    return serve_html("index.html")


@app.get("/api/gamemodes")
async def api_gamemodes():
    with db.session() as sess:
        from sqlalchemy import func
        rows = sess.query(ReplayRow.gamemode).distinct().order_by(ReplayRow.gamemode).all()
    return [r[0] for r in rows]


@app.get("/api/summary")
async def api_summary(gamemode: str = Query("blitz")):
    return get_summary(db, gamemode)


@app.get("/api/trends")
async def api_trends(gamemode: str = Query("blitz")):
    return get_trends(db, gamemode)


@app.get("/api/sessions")
async def api_sessions(gamemode: str = Query("blitz")):
    return get_session_summaries(db, gamemode)


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


@app.get("/api/replays")
async def api_replays(
    gamemode: Optional[str] = Query(None),
    limit: int = Query(100),
    offset: int = Query(0),
):
    rows = db.get_all_replays(gamemode=gamemode, limit=limit, offset=offset)
    data = []
    for replay, stats in rows:
        d = {"id": replay.id, "gamemode": replay.gamemode, "timestamp": replay.timestamp.isoformat(), "username": replay.username, "stats": {}}
        if stats:
            d["stats"] = {
                "score": stats.score, "lines": stats.lines, "apm": stats.apm, "pps": stats.pps,
                "level": stats.level, "kpp": stats.kpp, "kps": stats.kps,
                "final_time": stats.final_time,
                "finesse_faults": stats.finesse_faults,
                "finesse_perfect_pieces": stats.finesse_perfect_pieces,
                "pieces_placed": stats.pieces_placed,
                "quads": stats.quads,
                "all_clears": stats.all_clears,
            }
        data.append(d)
    return data


@app.get("/api/replays/{replay_id}")
async def api_replay_detail(replay_id: str):
    detail = get_replay_detail(db, replay_id)
    if not detail:
        return JSONResponse({"error": "not found"}, status_code=404)
    return detail


@app.get("/replays", response_class=HTMLResponse)
async def replays_page():
    return serve_html("replays.html")


@app.get("/replays/{replay_id}", response_class=HTMLResponse)
async def replay_detail_page(replay_id: str):
    detail = get_replay_detail(db, replay_id)
    if not detail:
        return HTMLResponse("Replay not found", status_code=404)

    html = serve_html("replay_detail.html")
    import json
    html = html.replace("/*{{REPLAY_DATA}}*/", json.dumps(detail))
    return html


@app.get("/trends", response_class=HTMLResponse)
async def trends_page():
    return serve_html("trends.html")


@app.get("/sessions", response_class=HTMLResponse)
async def sessions_page():
    return serve_html("sessions.html")


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


@app.get("/matches", response_class=HTMLResponse)
async def matches_page():
    return serve_html("matches.html")


@app.get("/matches/{match_id}", response_class=HTMLResponse)
async def match_detail_page(match_id: str):
    detail = db.get_match_detail(match_id)
    if not detail:
        return HTMLResponse("Match not found", status_code=404)
    html = serve_html("match_detail.html")
    import json

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


@app.get("/pbs", response_class=HTMLResponse)
async def pbs_page():
    return serve_html("pbs.html")
