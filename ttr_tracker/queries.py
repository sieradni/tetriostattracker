from datetime import datetime
from typing import Optional

from ttr_tracker.database import Database, ReplayRow, StatsRow, PartialRunRow
from ttr_tracker.survival import compute_likelihoods


def get_summary(db: Database, gamemode: str = "blitz") -> dict:
    total = db.count_replays(gamemode) + db.count_partial_runs(gamemode)
    if total == 0:
        return {"total": 0, "recent_avg": {}, "pbs": {}}

    entries = _combined_recent_entries(db, gamemode, 20)
    avg = _compute_averages(entries)

    pbs = _combined_pbs(db, gamemode)
    sessions = _combined_session_groups(db, gamemode)

    return {
        "total": total,
        "recent_avg": avg,
        "pbs": pbs,
        "session_count": len(sessions),
    }


def _combined_recent_entries(db: Database, gamemode: str, limit: Optional[int] = 20) -> list[dict]:
    with db.session() as sess:
        replay_q = (
            sess.query(ReplayRow, StatsRow)
            .outerjoin(StatsRow, ReplayRow.id == StatsRow.replay_id)
            .filter(ReplayRow.gamemode == gamemode)
            .order_by(ReplayRow.timestamp.desc())
        )
        if limit:
            replay_q = replay_q.limit(limit)
        replay_q = replay_q.all()

        partial_q = (
            sess.query(PartialRunRow)
            .filter(PartialRunRow.gamemode == gamemode)
            .order_by(PartialRunRow.timestamp.desc())
        )
        if limit:
            partial_q = partial_q.limit(limit)
        partial_q = partial_q.all()

    entries = []
    for rp, st in replay_q:
        entries.append({
            "type": "replay",
            "timestamp": rp.timestamp,
            "score": st.score if st else 0,
            "lines": st.lines if st else 0,
            "apm": st.apm if st else 0.0,
            "pps": st.pps if st else 0.0,
            "kpp": st.kpp if st else 0.0,
            "kps": st.kps if st else 0.0,
            "level": st.level if st else 0,
            "tspins": st.tspins if st else 0,
            "finesse_faults": st.finesse_faults if st else 0,
            "finesse_perfect_pieces": st.finesse_perfect_pieces if st else 0,
            "pieces_placed": st.pieces_placed if st else 0,
            "quads": st.quads if st else 0,
            "all_clears": st.all_clears if st else 0,
            "time": (st.final_time if st else 0) / 1000,
            "final_time": st.final_time if st else 0,
            "perfect_pct": (st.finesse_perfect_pieces / max(st.pieces_placed, 1)) if (st and st.pieces_placed) else 0,
        })
    for pr in partial_q:
        entries.append({
            "type": "partial",
            "timestamp": pr.timestamp,
            "score": pr.score,
            "lines": None,
            "apm": None,
            "pps": pr.pps,
            "kpp": pr.kpp,
            "kps": pr.kps,
            "level": None,
            "tspins": None,
            "finesse_faults": None,
            "finesse_perfect_pieces": None,
            "pieces_placed": pr.pieces_placed,
            "quads": None,
            "all_clears": pr.all_clears,
            "time": pr.time_elapsed,
            "final_time": pr.time_elapsed * 1000,
            "perfect_pct": None,
        })
    entries.sort(key=lambda e: e["timestamp"], reverse=True)
    return entries[:limit]


def _compute_averages(entries: list[dict]) -> dict:
    if not entries:
        return {}
    n = len(entries)

    def _avg(key, skip_none=True):
        vals = [e[key] for e in entries if not skip_none or e[key] is not None]
        if not vals:
            return 0.0
        return sum(vals) / len(vals)

    return {
        "score": _avg("score"),
        "lines": _avg("lines"),
        "apm": _avg("apm"),
        "pps": _avg("pps"),
        "level": _avg("level"),
        "tspins": _avg("tspins"),
        "kpp": _avg("kpp"),
        "kps": _avg("kps"),
        "time": _avg("time"),
        "finesse_faults": _avg("finesse_faults"),
        "perfect_pct": _avg("perfect_pct"),
    }


def get_trends(db: Database, gamemode: str = "blitz") -> dict[str, list[tuple[str, float]]]:
    columns = [
        "score", "lines", "apm", "pps", "level", "tspins", "finesse_faults",
        "kpp", "kps", "quads", "all_clears", "final_time", "pieces_placed",
    ]
    raw = db.get_trend_data_multi(gamemode, columns)
    result: dict[str, list[tuple[str, float]]] = {}
    for col in columns:
        pairs = raw.get(col, [])
        if col == "final_time":
            result["time"] = [(ts.isoformat(), val / 1000) for ts, val in pairs if val]
        else:
            result[col] = [(ts.isoformat(), val) for ts, val in pairs]

    # score_per_piece is derived from the same batched data
    score_pairs = raw.get("score", [])
    pieces_pairs = raw.get("pieces_placed", [])
    result["score_per_piece"] = [
        (ts.isoformat(), s / max(p, 1))
        for (ts, s), (_, p) in zip(score_pairs, pieces_pairs)
    ]

    return result


def get_session_summaries(db: Database, gamemode: str = "blitz") -> list[dict]:
    groups = _combined_session_groups(db, gamemode)
    summaries = []
    for group in groups:
        start = group[0]["timestamp"]
        end = group[-1]["timestamp"]
        duration = (end - start).total_seconds() / 60
        count = len(group)
        scores = [e["score"] for e in group if e["score"] is not None]
        times = [e["time"] for e in group if e["time"] is not None]
        avg_score = sum(scores) / len(scores) if scores else 0
        best_score = max(scores) if scores else 0
        avg_time = sum(times) / len(times) if times else 0
        best_time = min(times) if times else 0
        summaries.append({
            "start": start.isoformat(),
            "end": end.isoformat(),
            "duration_min": round(duration, 1),
            "count": count,
            "avg_score": round(avg_score),
            "best_score": best_score,
            "avg_time": round(avg_time, 2),
            "best_time": round(best_time, 2),
        })
    return summaries


def _combined_session_groups(db: Database, gamemode: str, gap_minutes: int = 60) -> list[list[dict]]:
    entries = _combined_recent_entries(db, gamemode, limit=None)
    entries.sort(key=lambda e: e["timestamp"])
    if not entries:
        return []
    groups: list[list[dict]] = [[entries[0]]]
    for e in entries[1:]:
        gap = (e["timestamp"] - groups[-1][-1]["timestamp"]).total_seconds() / 60
        if gap > gap_minutes:
            groups.append([])
        groups[-1].append(e)
    return groups


def _combined_pbs(db: Database, gamemode: str) -> dict[str, tuple[float, str, datetime]]:
    result = {}
    _add_replay_pbs(db, gamemode, result)
    _add_partial_pbs(db, gamemode, result)
    return result


def _add_replay_pbs(db: Database, gamemode: str, result: dict) -> None:
    pb_columns = [
        "score", "lines", "apm", "pps", "level", "top_combo", "top_btb",
        "tspins", "quads", "all_clears", "finesse_perfect_pieces",
        "kpp", "kps", "final_time",
    ]
    with db.session() as sess:
        rows = (
            sess.query(StatsRow, ReplayRow.id, ReplayRow.timestamp)
            .join(ReplayRow, ReplayRow.id == StatsRow.replay_id)
            .filter(ReplayRow.gamemode == gamemode)
            .all()
        )
        for st, rid, ts in rows:
            for col_name in pb_columns:
                val = getattr(st, col_name, None)
                if val is None:
                    continue
                existing = result.get(col_name)
                if col_name == "final_time":
                    if existing is None or val < existing[0]:
                        result[col_name] = (float(val), rid, ts)
                else:
                    if existing is None or val > existing[0]:
                        result[col_name] = (float(val), rid, ts)


def _add_partial_pbs(db: Database, gamemode: str, result: dict) -> None:
    partials = db.get_all_partial_runs(gamemode=gamemode, limit=None)
    for pr in partials:
        _check_pb(result, "score", float(pr.score), f"partial-{pr.id}", pr.timestamp, higher=True)
        _check_pb(result, "pps", float(pr.pps), f"partial-{pr.id}", pr.timestamp, higher=True)
        _check_pb(result, "kpp", float(pr.kpp), f"partial-{pr.id}", pr.timestamp, higher=True)
        _check_pb(result, "kps", float(pr.kps), f"partial-{pr.id}", pr.timestamp, higher=True)
        _check_pb(result, "all_clears", float(pr.all_clears), f"partial-{pr.id}", pr.timestamp, higher=True)
        _check_pb(result, "pieces_placed", float(pr.pieces_placed), f"partial-{pr.id}", pr.timestamp, higher=True)
        if pr.time_elapsed:
            _check_pb(result, "final_time", pr.time_elapsed * 1000, f"partial-{pr.id}", pr.timestamp, higher=False)


def _check_pb(result: dict, key: str, value: float, eid: str, ts: datetime, higher: bool) -> None:
    existing = result.get(key)
    if existing is None:
        result[key] = (value, eid, ts)
    elif higher and value > existing[0]:
        result[key] = (value, eid, ts)
    elif not higher and value < existing[0]:
        result[key] = (value, eid, ts)


def get_replay_detail(db: Database, replay_id: str) -> Optional[dict]:
    result = db.get_replay(replay_id)
    if not result:
        return None
    replay_row, stats_row, options_row = result

    data = {
        "id": replay_row.id,
        "gamemode": replay_row.gamemode,
        "timestamp": replay_row.timestamp.isoformat(),
        "username": replay_row.username,
        "options": {},
        "stats": {},
    }
    if options_row:
        data["options"] = {
            "mission": options_row.mission,
            "objective_type": options_row.objective_type,
            "objective_time": options_row.objective_time,
            "handling_arr": options_row.handling_arr,
            "handling_das": options_row.handling_das,
        }
    if stats_row:
        data["stats"] = {
            "score": stats_row.score,
            "lines": stats_row.lines,
            "level": stats_row.level,
            "apm": stats_row.apm,
            "pps": stats_row.pps,
            "pieces_placed": stats_row.pieces_placed,
            "inputs": stats_row.inputs,
            "holds": stats_row.holds,
            "top_combo": stats_row.top_combo,
            "top_btb": stats_row.top_btb,
            "tspins": stats_row.tspins,
            "singles": stats_row.singles,
            "doubles": stats_row.doubles,
            "triples": stats_row.triples,
            "quads": stats_row.quads,
            "pentas": stats_row.pentas,
            "tspin_singles": stats_row.tspin_singles,
            "tspin_doubles": stats_row.tspin_doubles,
            "tspin_triples": stats_row.tspin_triples,
            "tspin_quads": stats_row.tspin_quads,
            "all_clears": stats_row.all_clears,
            "kpp": stats_row.kpp,
            "kps": stats_row.kps,
            "finesse_combo": stats_row.finesse_combo,
            "finesse_faults": stats_row.finesse_faults,
            "finesse_perfect_pieces": stats_row.finesse_perfect_pieces,
            "final_time": stats_row.final_time,
            "gameover_reason": stats_row.gameover_reason,
        }
    return data


def get_partial_run_detail(db: Database, run_id: int) -> Optional[dict]:
    pr = db.get_partial_run(run_id)
    if not pr:
        return None
    return {
        "id": f"partial-{pr.id}",
        "type": "partial",
        "gamemode": pr.gamemode,
        "timestamp": pr.timestamp.isoformat(),
        "username": None,
        "stats": {
            "score": pr.score,
            "pieces_placed": pr.pieces_placed,
            "pps": pr.pps,
            "inputs": pr.inputs,
            "kpp": pr.kpp,
            "kps": pr.kps,
            "spp": pr.spp,
            "all_clears": pr.all_clears,
            "time_left": pr.time_left,
            "final_time": pr.time_elapsed * 1000,
            "gameover_reason": "misdrop",
        },
        "source_image": pr.source_image,
        "notes": pr.notes,
    }


def get_blitz_survival(db: Database, gamemode: str = "blitz",
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None) -> dict:
    full, partial = db.get_all_misdrop_data(gamemode, start_date, end_date)
    return compute_likelihoods(full, partial)


def get_match_leaderboard(db: Database, player_id: str) -> dict:
    raw = db.get_match_aggregate_trends(player_id)
    if not raw:
        return {"total_matches": 0, "total_wins": 0, "win_rate": 0, "avg_apm": 0, "avg_pps": 0, "avg_vsscore": 0, "total_kills": 0}
    total_matches = len(raw)
    total_match_wins = sum(1 for r in raw if r["wins"] > r.get("opponent_rounds_won", 0))
    total_round_wins = sum(r["wins"] for r in raw)
    total_rounds = sum(r["rounds"] for r in raw)
    avg_apm = sum(r["avg_apm"] for r in raw) / total_matches
    avg_pps = sum(r["avg_pps"] for r in raw) / total_matches
    avg_vsscore = sum(r["avg_vsscore"] for r in raw) / total_matches
    total_kills = sum(r["total_kills"] for r in raw)
    return {
        "total_matches": total_matches,
        "total_wins": total_match_wins,
        "total_round_wins": total_round_wins,
        "total_rounds": total_rounds,
        "round_win_rate": round(total_round_wins / total_rounds * 100, 1) if total_rounds else 0,
        "match_win_rate": round(total_match_wins / total_matches * 100, 1) if total_matches else 0,
        "avg_apm": round(avg_apm, 1),
        "avg_pps": round(avg_pps, 2),
        "avg_vsscore": round(avg_vsscore, 1),
        "total_kills": total_kills,
    }


def get_match_trends(db: Database, player_name: str) -> dict[str, list[tuple[str, float]]]:
    columns = ["apm", "pps", "vsscore", "kills", "garbagesent"]
    result = {}
    for col in columns:
        raw = db.get_match_trend_data(col, player_name)
        result[col] = [(ts.isoformat(), val) for ts, val in raw]
    return result
