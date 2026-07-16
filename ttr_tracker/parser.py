import json
from pathlib import Path

from ttr_tracker.models import TTRFile


class ParseError(ValueError):
    """Raised when a .ttr or .ttrm file cannot be parsed."""


def parse_ttr(path: str | Path) -> TTRFile:
    try:
        raw = Path(path).read_text(encoding="utf-8")
        data = json.loads(raw)
        return TTRFile.model_validate(data)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
        raise ParseError(f"Failed to parse .ttr file {path}: {e}") from e


def parse_ttrm(path: str | Path) -> dict:
    try:
        raw = Path(path).read_text(encoding="utf-8")
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
        raise ParseError(f"Failed to parse .ttrm file {path}: {e}") from e

    replay = data.get("replay", {})
    leaderboard = replay.get("leaderboard", [])
    rounds_raw = replay.get("rounds", [])

    rounds = []
    for r_idx, round_data in enumerate(rounds_raw):
        for player in round_data:
            ps = player.get("stats", {})
            rounds.append({
                "round_number": r_idx + 1,
                "player_id": player.get("id"),
                "player_name": player.get("username"),
                "alive": player.get("alive", False),
                "lifetime": player.get("lifetime", 0),
                "apm": ps.get("apm", 0),
                "pps": ps.get("pps", 0),
                "vsscore": ps.get("vsscore", 0),
                "garbagesent": ps.get("garbagesent", 0),
                "garbagereceived": ps.get("garbagereceived", 0),
                "kills": ps.get("kills", 0),
                "btb": ps.get("btb", 0),
                "revives": ps.get("revives", 0),
            })

    leaderboard_data = []
    for lb in leaderboard:
        ls = lb.get("stats", {})
        leaderboard_data.append({
            "player_id": lb.get("id"),
            "player_name": lb.get("username"),
            "wins": lb.get("wins", 0),
            "apm": ls.get("apm", 0),
            "pps": ls.get("pps", 0),
            "vsscore": ls.get("vsscore", 0),
            "garbagesent": ls.get("garbagesent", 0),
            "garbagereceived": ls.get("garbagereceived", 0),
            "kills": ls.get("kills", 0),
        })

    return {
        "id": data.get("id"),
        "gamemode": data.get("gamemode", "league"),
        "ts": data.get("ts"),
        "users": data.get("users", []),
        "leaderboard": leaderboard_data,
        "rounds": rounds,
    }
