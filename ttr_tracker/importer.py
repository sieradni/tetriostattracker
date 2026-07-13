from datetime import datetime
from pathlib import Path
from ttr_tracker.database import Database
from ttr_tracker.parser import parse_ttr, parse_ttrm


def import_file(db: Database, path: str | Path) -> str:
    p = Path(path)
    if p.suffix == ".ttrm":
        return import_ttrm_file(db, p)
    ttr = parse_ttr(path)

    if db.replay_exists(ttr.id):
        return f"Skipped {p.name} (already imported)"

    user = ttr.users[0] if ttr.users else None

    db.insert_replay(
        replay_id=ttr.id,
        gamemode=ttr.gamemode,
        timestamp=ttr.ts,
        username=user.username if user else None,
        user_id=user.id if user else None,
        version=ttr.version,
    )

    s = ttr.replay.results.stats
    a = ttr.replay.results.aggregatestats
    db.insert_stats(
        replay_id=ttr.id,
        gamemode=ttr.gamemode,
        apm=a.apm,
        pps=a.pps,
        vsscore=a.vsscore,
        score=s.score,
        lines=s.lines,
        level=s.level,
        pieces_placed=s.piecesplaced,
        inputs=s.inputs,
        holds=s.holds,
        top_combo=s.topcombo,
        top_btb=s.topbtb,
        tspins=s.tspins,
        singles=s.clears.singles,
        doubles=s.clears.doubles,
        triples=s.clears.triples,
        quads=s.clears.quads,
        pentas=s.clears.pentas,
        tspin_singles=s.clears.tspinsingles,
        tspin_doubles=s.clears.tspindoubles,
        tspin_triples=s.clears.tspintriples,
        tspin_quads=s.clears.tspinquads,
        all_clears=s.clears.allclear,
        finesse_combo=s.finesse.combo,
        finesse_faults=s.finesse.faults,
        finesse_perfect_pieces=s.finesse.perfectpieces,
        garbage_sent=s.garbage.sent,
        garbage_received=s.garbage.received,
        kpp=s.inputs / s.piecesplaced if s.piecesplaced else 0,
        kps=s.inputs / (s.finaltime / 1000) if s.finaltime else 0,
        final_time=s.finaltime,
        gameover_reason=ttr.replay.results.gameoverreason,
    )

    o = ttr.replay.options
    db.insert_options(
        replay_id=ttr.id,
        objective_type=o.objective_type,
        objective_time=o.objective_time,
        mission=o.mission,
        handling_arr=o.handling.arr,
        handling_das=o.handling.das,
        handling_sdf=o.handling.sdf,
    )

    return f"Imported {p.name} ({ttr.gamemode}, {s.score} pts, {s.lines} lines)"


def import_ttrm_file(db: Database, path: str | Path) -> str:
    data = parse_ttrm(path)
    match_id = data["id"]

    if db.match_exists(match_id):
        return f"Skipped {Path(path).name} (already imported)"

    ts_str = data.get("ts")
    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")) if ts_str else datetime.now()

    lb = data.get("leaderboard", [])
    p1 = lb[0] if len(lb) > 0 else {}
    p2 = lb[1] if len(lb) > 1 else {}

    db.insert_match(
        match_id=match_id,
        timestamp=ts,
        player1_id=p1.get("player_id", ""),
        player1_name=p1.get("player_name", ""),
        player2_id=p2.get("player_id", ""),
        player2_name=p2.get("player_name", ""),
        player1_wins=p1.get("wins", 0),
        player2_wins=p2.get("wins", 0),
    )

    for rd in data.get("rounds", []):
        db.insert_match_round(
            match_id=match_id,
            round_number=rd["round_number"],
            player_id=rd["player_id"],
            player_name=rd["player_name"],
            alive=1 if rd.get("alive") else 0,
            lifetime=rd.get("lifetime", 0),
            apm=rd.get("apm", 0),
            pps=rd.get("pps", 0),
            vsscore=rd.get("vsscore", 0),
            garbagesent=rd.get("garbagesent", 0),
            garbagereceived=rd.get("garbagereceived", 0),
            kills=rd.get("kills", 0),
            btb=rd.get("btb", 0),
            revives=rd.get("revives", 0),
        )

    return f"Imported {Path(path).name} (league, {p1.get('player_name','?')} {p1.get('wins',0)}-{p2.get('wins',0)} {p2.get('player_name','?')})"


def import_directory(db: Database, directory: str | Path) -> list[str]:
    results = []
    for fpath in sorted(Path(directory).iterdir()):
        if fpath.suffix in (".ttr", ".ttrm"):
            results.append(import_file(db, fpath))
    return results
