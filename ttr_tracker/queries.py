from datetime import datetime, timedelta
from typing import Optional

from ttr_tracker.database import Database, ReplayRow, StatsRow


def get_summary(db: Database, gamemode: str = "blitz") -> dict:
    total = db.count_replays(gamemode)
    if total == 0:
        return {"total": 0, "recent_avg": {}, "pbs": {}}

    with db.session() as sess:
        recent = (
            sess.query(StatsRow)
            .join(ReplayRow, ReplayRow.id == StatsRow.replay_id)
            .filter(ReplayRow.gamemode == gamemode)
            .order_by(ReplayRow.timestamp.desc())
            .limit(20)
            .all()
        )

    avg = {}
    if recent:
        n = len(recent)
        avg = {
            "score": sum(r.score or 0 for r in recent) / n,
            "lines": sum(r.lines or 0 for r in recent) / n,
            "apm": sum(r.apm or 0 for r in recent) / n,
            "pps": sum(r.pps or 0 for r in recent) / n,
            "level": sum(r.level or 0 for r in recent) / n,
            "tspins": sum(r.tspins or 0 for r in recent) / n,
            "kpp": sum(r.kpp or 0 for r in recent) / n,
            "kps": sum(r.kps or 0 for r in recent) / n,
            "time": sum((r.final_time or 0) / 1000 for r in recent) / n,
            "finesse_faults": sum(r.finesse_faults or 0 for r in recent) / n,
            "perfect_pct": sum((r.finesse_perfect_pieces or 0) / max(r.pieces_placed or 1, 1) for r in recent) / n,
        }

    pbs = db.get_pbs(gamemode)
    sessions = db.get_session_groups(gamemode)

    return {
        "total": total,
        "recent_avg": avg,
        "pbs": pbs,
        "session_count": len(sessions),
    }


def get_trends(db: Database, gamemode: str = "blitz") -> dict[str, list[tuple[str, float]]]:
    columns = ["score", "lines", "apm", "pps", "level", "tspins", "finesse_faults", "kpp", "kps", "quads", "all_clears"]
    result = {}
    for col in columns:
        raw = db.get_trend_data(gamemode, col)
        result[col] = [(ts.isoformat(), val) for ts, val in raw]
    time_raw = db.get_trend_data(gamemode, "final_time")
    result["time"] = [(ts.isoformat(), val / 1000) for ts, val in time_raw if val]

    pieces_raw = db.get_trend_data(gamemode, "pieces_placed")
    result["score_per_piece"] = [
        (ts, s / max(p, 1))
        for (ts, s), (_, p) in zip(result["score"], pieces_raw)
    ]
    return result


def get_session_summaries(db: Database, gamemode: str = "blitz") -> list[dict]:
    groups = db.get_session_groups(gamemode)
    summaries = []
    for group in groups:
        ids = [r.id for r in group]
        with db.session() as sess:
            stats = (
                sess.query(StatsRow)
                .filter(StatsRow.replay_id.in_(ids))
                .all()
            )
        start = group[0].timestamp
        end = group[-1].timestamp
        duration = (end - start).total_seconds() / 60
        count = len(group)
        avg_score = sum(s.score or 0 for s in stats) / count if stats else 0
        best_score = max((s.score or 0 for s in stats), default=0)
        avg_time = sum((s.final_time or 0) / 1000 for s in stats) / count if stats else 0
        best_time = min((s.final_time or 0) / 1000 for s in stats) if stats else 0
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
