from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import (
    Column,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    DateTime,
    ForeignKey,
    NullPool,
)
from sqlalchemy.orm import Session, declarative_base, relationship

Base = declarative_base()


class ReplayRow(Base):
    __tablename__ = "replays"

    id = Column(String, primary_key=True)
    gamemode = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    user_id = Column(String, nullable=True)
    username = Column(String, nullable=True)
    version = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))

    stats = relationship("StatsRow", back_populates="replay", uselist=False, cascade="all, delete-orphan")
    options = relationship("OptionsRow", back_populates="replay", uselist=False, cascade="all, delete-orphan")


class StatsRow(Base):
    __tablename__ = "replay_stats"

    replay_id = Column(String, ForeignKey("replays.id"), primary_key=True)
    gamemode = Column(String, nullable=False, index=True)

    apm = Column(Float, default=0.0)
    pps = Column(Float, default=0.0)
    vsscore = Column(Float, default=0.0)

    score = Column(Integer, default=0)
    lines = Column(Integer, default=0)
    level = Column(Integer, default=1)
    pieces_placed = Column(Integer, default=0)
    inputs = Column(Integer, default=0)
    holds = Column(Integer, default=0)
    top_combo = Column(Integer, default=0)
    top_btb = Column(Integer, default=1)
    tspins = Column(Integer, default=0)

    singles = Column(Integer, default=0)
    doubles = Column(Integer, default=0)
    triples = Column(Integer, default=0)
    quads = Column(Integer, default=0)
    pentas = Column(Integer, default=0)
    tspin_singles = Column(Integer, default=0)
    tspin_doubles = Column(Integer, default=0)
    tspin_triples = Column(Integer, default=0)
    tspin_quads = Column(Integer, default=0)
    all_clears = Column(Integer, default=0)

    finesse_combo = Column(Integer, default=0)
    finesse_faults = Column(Integer, default=0)
    finesse_perfect_pieces = Column(Integer, default=0)

    garbage_sent = Column(Integer, default=0)
    garbage_received = Column(Integer, default=0)

    kpp = Column(Float, default=0.0)
    kps = Column(Float, default=0.0)

    final_time = Column(Float, default=0.0)
    gameover_reason = Column(String, nullable=True)

    replay = relationship("ReplayRow", back_populates="stats")


class MatchRow(Base):
    __tablename__ = "matches"

    id = Column(String, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    gamemode = Column(String, default="league")
    player1_id = Column(String, nullable=True)
    player1_name = Column(String, nullable=True)
    player2_id = Column(String, nullable=True)
    player2_name = Column(String, nullable=True)
    player1_wins = Column(Integer, default=0)
    player2_wins = Column(Integer, default=0)
    winner_name = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))

    rounds = relationship("MatchRoundRow", back_populates="match", lazy="selectin")


class MatchRoundRow(Base):
    __tablename__ = "match_rounds"

    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(String, ForeignKey("matches.id"), nullable=False)
    round_number = Column(Integer, nullable=False)

    player_id = Column(String, nullable=True)
    player_name = Column(String, nullable=True)
    alive = Column(Integer, default=1)
    lifetime = Column(Integer, default=0)

    apm = Column(Float, default=0.0)
    pps = Column(Float, default=0.0)
    vsscore = Column(Float, default=0.0)
    garbagesent = Column(Integer, default=0)
    garbagereceived = Column(Integer, default=0)
    kills = Column(Integer, default=0)
    btb = Column(Integer, default=0)
    revives = Column(Integer, default=0)

    match = relationship("MatchRow", back_populates="rounds")


class PartialRunRow(Base):
    __tablename__ = "partial_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    gamemode = Column(String, nullable=False, default="blitz")
    score = Column(Integer, default=0)
    pieces_placed = Column(Integer, default=0)
    pps = Column(Float, default=0.0)
    inputs = Column(Integer, default=0)
    kpp = Column(Float, default=0.0)
    spp = Column(Float, default=0.0)
    all_clears = Column(Integer, default=0)
    time_left = Column(Float, default=0.0)
    time_elapsed = Column(Float, default=0.0)
    kps = Column(Float, default=0.0)
    source_image = Column(String, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))


class OptionsRow(Base):
    __tablename__ = "replay_options"

    replay_id = Column(String, ForeignKey("replays.id"), primary_key=True)
    objective_type = Column(String, nullable=True)
    objective_time = Column(Integer, nullable=True)
    mission = Column(String, nullable=True)
    handling_arr = Column(Integer, nullable=True)
    handling_das = Column(Integer, nullable=True)
    handling_sdf = Column(Integer, nullable=True)

    replay = relationship("ReplayRow", back_populates="options")


class Database:
    def __init__(self, db_path: str | Path) -> None:
        if str(db_path) == ":memory:":
            url = "sqlite:///:memory:"
            poolclass = None
            connect_args = {}
        else:
            path = Path(db_path).resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            url = "sqlite:///" + path.as_posix()
            poolclass = NullPool
            connect_args = {"check_same_thread": False}
        self.engine = create_engine(
            url,
            echo=False,
            poolclass=poolclass,
            connect_args=connect_args,
        )
        Base.metadata.create_all(self.engine)

    @staticmethod
    def _to_utc_naive(dt: datetime) -> datetime:
        """Convert a datetime to a naive UTC datetime for storage."""
        if dt.tzinfo is not None:
            return dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        sess = Session(self.engine, expire_on_commit=False)
        try:
            yield sess
            sess.commit()
        except:
            sess.rollback()
            raise
        finally:
            sess.close()

    def replay_exists(self, replay_id: str) -> bool:
        with self.session() as sess:
            return sess.query(ReplayRow).filter_by(id=replay_id).first() is not None

    def insert_replay(
        self,
        replay_id: str,
        gamemode: str,
        timestamp: datetime,
        username: Optional[str],
        user_id: Optional[str],
        version: int,
    ) -> None:
        with self.session() as sess:
            row = ReplayRow(
                id=replay_id,
                gamemode=gamemode,
                timestamp=self._to_utc_naive(timestamp),
                username=username,
                user_id=user_id,
                version=version,
            )
            sess.add(row)

    def insert_stats(
        self,
        replay_id: str,
        gamemode: str,
        **kwargs,
    ) -> None:
        with self.session() as sess:
            row = StatsRow(replay_id=replay_id, gamemode=gamemode, **kwargs)
            sess.add(row)

    def insert_options(
        self,
        replay_id: str,
        **kwargs,
    ) -> None:
        with self.session() as sess:
            row = OptionsRow(replay_id=replay_id, **kwargs)
            sess.add(row)

    def count_replays(self, gamemode: Optional[str] = None) -> int:
        with self.session() as sess:
            q = sess.query(ReplayRow)
            if gamemode:
                q = q.filter_by(gamemode=gamemode)
            return q.count()

    def get_all_replays(
        self, gamemode: Optional[str] = None, limit: int = 100, offset: int = 0
    ) -> list[tuple[ReplayRow, Optional[StatsRow]]]:
        with self.session() as sess:
            q = sess.query(ReplayRow, StatsRow).outerjoin(
                StatsRow, ReplayRow.id == StatsRow.replay_id
            )
            if gamemode:
                q = q.filter(ReplayRow.gamemode == gamemode)
            q = q.order_by(ReplayRow.timestamp.desc()).limit(limit).offset(offset)
            return list(q.all())

    def get_replay(self, replay_id: str) -> Optional[tuple[ReplayRow, Optional[StatsRow], Optional[OptionsRow]]]:
        with self.session() as sess:
            result = (
                sess.query(ReplayRow, StatsRow, OptionsRow)
                .outerjoin(StatsRow, ReplayRow.id == StatsRow.replay_id)
                .outerjoin(OptionsRow, ReplayRow.id == OptionsRow.replay_id)
                .filter(ReplayRow.id == replay_id)
                .first()
            )
            return result

    def get_trend_data(
        self, gamemode: str, stat_column: str, limit: int = 200
    ) -> list[tuple[datetime, float]]:
        with self.session() as sess:
            col = getattr(StatsRow, stat_column, None)
            if col is None:
                return []
            q = (
                sess.query(ReplayRow.timestamp, col)
                .join(StatsRow, ReplayRow.id == StatsRow.replay_id)
                .filter(ReplayRow.gamemode == gamemode)
                .order_by(ReplayRow.timestamp.asc())
                .limit(limit)
            )
            return [(ts, float(v) if v is not None else 0.0) for ts, v in q.all()]

    def get_trend_data_pair(
        self, gamemode: str, col1: str, col2: str, limit: int = 200
    ) -> list[tuple[datetime, float, float]]:
        """Return two stat columns in a single query so rows are guaranteed aligned."""
        with self.session() as sess:
            c1 = getattr(StatsRow, col1, None)
            c2 = getattr(StatsRow, col2, None)
            if c1 is None or c2 is None:
                return []
            q = (
                sess.query(ReplayRow.timestamp, c1, c2)
                .join(StatsRow, ReplayRow.id == StatsRow.replay_id)
                .filter(ReplayRow.gamemode == gamemode)
                .order_by(ReplayRow.timestamp.asc())
                .limit(limit)
            )
            return [
                (ts, float(v1) if v1 is not None else 0.0, float(v2) if v2 is not None else 0.0)
                for ts, v1, v2 in q.all()
            ]

    def get_trend_data_multi(
        self, gamemode: str, columns: list[str], limit: int = 200
    ) -> dict[str, list[tuple[datetime, float]]]:
        """Return multiple stat columns in a single batched query.

        Returns dict mapping column name -> list of (timestamp, value) pairs.
        All lists are aligned to the same rows (same ordering, same length).
        """
        attrs = []
        for col in columns:
            c = getattr(StatsRow, col, None)
            if c is None:
                continue
            attrs.append((col, c))

        if not attrs:
            return {}

        with self.session() as sess:
            entities = [ReplayRow.timestamp] + [ac for _, ac in attrs]
            q = (
                sess.query(*entities)
                .join(StatsRow, ReplayRow.id == StatsRow.replay_id)
                .filter(ReplayRow.gamemode == gamemode)
                .order_by(ReplayRow.timestamp.asc())
                .limit(limit)
            )
            rows = q.all()
            result: dict[str, list[tuple[datetime, float]]] = {col: [] for col, _ in attrs}
            for row in rows:
                ts = row[0]
                for i, (col, _) in enumerate(attrs):
                    val = row[i + 1]
                    result[col].append((ts, float(val) if val is not None else 0.0))
            return result

    def match_exists(self, match_id: str) -> bool:
        with self.session() as sess:
            return sess.query(MatchRow).filter_by(id=match_id).first() is not None

    def insert_match(
        self,
        match_id: str,
        timestamp: datetime,
        player1_id: str,
        player1_name: str,
        player2_id: str,
        player2_name: str,
        player1_wins: int,
        player2_wins: int,
    ) -> None:
        winner_name = player1_name if player1_wins > player2_wins else player2_name
        with self.session() as sess:
            row = MatchRow(
                id=match_id,
                timestamp=self._to_utc_naive(timestamp),
                player1_id=player1_id,
                player1_name=player1_name,
                player2_id=player2_id,
                player2_name=player2_name,
                player1_wins=player1_wins,
                player2_wins=player2_wins,
                winner_name=winner_name,
            )
            sess.add(row)

    def insert_match_round(self, match_id: str, round_number: int, **kwargs) -> None:
        with self.session() as sess:
            row = MatchRoundRow(match_id=match_id, round_number=round_number, **kwargs)
            sess.add(row)
            sess.commit()

    def get_all_matches(self, limit: int = 100, offset: int = 0) -> list[MatchRow]:
        with self.session() as sess:
            return (
                sess.query(MatchRow)
                .order_by(MatchRow.timestamp.desc())
                .limit(limit)
                .offset(offset)
                .all()
            )

    def get_match_detail(self, match_id: str):
        with self.session() as sess:
            match = sess.query(MatchRow).filter_by(id=match_id).first()
            if not match:
                return None
            rounds = (
                sess.query(MatchRoundRow)
                .filter_by(match_id=match_id)
                .order_by(MatchRoundRow.round_number, MatchRoundRow.player_name)
                .all()
            )
            return match, rounds

    def get_match_trend_data(
        self, stat_column: str, player_name: str, limit: int = 200
    ) -> list[tuple[datetime, float]]:
        with self.session() as sess:
            col = getattr(MatchRoundRow, stat_column, None)
            if col is None:
                return []
            match_ids = (
                sess.query(MatchRow.id)
                .join(MatchRoundRow, MatchRow.id == MatchRoundRow.match_id)
                .filter(MatchRoundRow.player_name == player_name)
                .distinct()
                .order_by(MatchRow.timestamp.asc())
                .limit(limit)
                .all()
            )
            match_ids = [m[0] for m in match_ids]
            if not match_ids:
                return []

            rows = (
                sess.query(MatchRow.timestamp, col)
                .join(MatchRoundRow, MatchRow.id == MatchRoundRow.match_id)
                .filter(
                    MatchRoundRow.match_id.in_(match_ids),
                    MatchRoundRow.player_name == player_name,
                )
                .order_by(MatchRow.timestamp.asc(), MatchRoundRow.round_number.asc())
                .all()
            )
            return [(ts, float(v) if v is not None else 0.0) for ts, v in rows]

    def get_match_aggregate_trends(
        self, player_id: Optional[str], limit: int = 200
    ) -> list[dict]:
        if not player_id:
            return []
        with self.session() as sess:
            matches = (
                sess.query(MatchRow)
                .filter(
                    (MatchRow.player1_id == player_id) | (MatchRow.player2_id == player_id)
                )
                .order_by(MatchRow.timestamp.desc())
                .limit(limit)
                .all()
            )
            matches = list(reversed(matches))  # back to chronological for trend display
            match_ids = [m.id for m in matches]
            all_rounds = (
                sess.query(MatchRoundRow)
                .filter(
                    MatchRoundRow.match_id.in_(match_ids),
                    MatchRoundRow.player_id == player_id,
                )
                .all()
            )
            rounds_by_match: dict[str, list[MatchRoundRow]] = {}
            for r in all_rounds:
                rounds_by_match.setdefault(r.match_id, []).append(r)
            result = []
            for m in matches:
                rounds = rounds_by_match.get(m.id, [])
                if not rounds:
                    continue
                avg_apm = sum(r.apm or 0 for r in rounds) / len(rounds)
                avg_pps = sum(r.pps or 0 for r in rounds) / len(rounds)
                avg_vsscore = sum(r.vsscore or 0 for r in rounds) / len(rounds)
                total_kills = sum(r.kills or 0 for r in rounds)
                wins = 0
                opponent_rounds_won = 0
                if player_id == m.player1_id:
                    wins = m.player1_wins
                    opponent_rounds_won = m.player2_wins
                elif player_id == m.player2_id:
                    wins = m.player2_wins
                    opponent_rounds_won = m.player1_wins
                result.append({
                    "ts": m.timestamp,
                    "match_id": m.id,
                    "wins": wins,
                    "opponent_rounds_won": opponent_rounds_won,
                    "rounds": len(rounds),
                    "avg_apm": avg_apm,
                    "avg_pps": avg_pps,
                    "avg_vsscore": avg_vsscore,
                    "total_kills": total_kills,
                    "opponent": m.player2_name if player_id == m.player1_id else m.player1_name,
                })
            return result

    # ── Partial Run CRUD ──────────────────────────────────────────────────

    def insert_partial_run(self, **kwargs) -> int:
        if "timestamp" in kwargs:
            kwargs["timestamp"] = self._to_utc_naive(kwargs["timestamp"])
        with self.session() as sess:
            row = PartialRunRow(**kwargs)
            sess.add(row)
            return row.id

    def get_all_partial_runs(
        self, gamemode: Optional[str] = None, limit: int = 100, offset: int = 0
    ) -> list[PartialRunRow]:
        with self.session() as sess:
            q = sess.query(PartialRunRow)
            if gamemode:
                q = q.filter(PartialRunRow.gamemode == gamemode)
            q = q.order_by(PartialRunRow.timestamp.desc()).offset(offset)
            if limit:
                q = q.limit(limit)
            return list(q.all())

    def get_partial_run(self, run_id: int) -> Optional[PartialRunRow]:
        with self.session() as sess:
            return sess.query(PartialRunRow).filter_by(id=run_id).first()

    def delete_partial_run(self, run_id: int) -> bool:
        with self.session() as sess:
            row = sess.query(PartialRunRow).filter_by(id=run_id).first()
            if not row:
                return False
            sess.delete(row)
            return True

    def delete_replay(self, replay_id: str) -> bool:
        with self.session() as sess:
            row = sess.query(ReplayRow).filter_by(id=replay_id).first()
            if not row:
                return False
            sess.delete(row)
            return True

    def update_partial_run(self, run_id: int, **kwargs) -> bool:
        if "timestamp" in kwargs:
            kwargs["timestamp"] = self._to_utc_naive(kwargs["timestamp"])
        with self.session() as sess:
            row = sess.query(PartialRunRow).filter_by(id=run_id).first()
            if not row:
                return False
            for k, v in kwargs.items():
                setattr(row, k, v)
            return True

    def count_partial_runs(self, gamemode: Optional[str] = None) -> int:
        with self.session() as sess:
            q = sess.query(PartialRunRow)
            if gamemode:
                q = q.filter(PartialRunRow.gamemode == gamemode)
            return q.count()

    def get_combined_entries(
        self, gamemode: Optional[str] = None, limit: int = 200
    ) -> list[dict]:
        """Return unified list of replay + partial-run entries sorted by timestamp desc."""
        entries = []
        with self.session() as sess:
            replay_q = (
                sess.query(ReplayRow, StatsRow)
                .outerjoin(StatsRow, ReplayRow.id == StatsRow.replay_id)
            )
            if gamemode:
                replay_q = replay_q.filter(ReplayRow.gamemode == gamemode)
            for rp, st in replay_q.all():
                entries.append({
                    "id": rp.id,
                    "type": "replay",
                    "gamemode": rp.gamemode,
                    "timestamp": rp.timestamp,
                    "username": rp.username,
                    "stats": {
                        "score": st.score if st else None,
                        "lines": st.lines if st else None,
                        "level": st.level if st else None,
                        "apm": st.apm if st else None,
                        "pps": st.pps if st else None,
                        "kpp": st.kpp if st else None,
                        "kps": st.kps if st else None,
                        "pieces_placed": st.pieces_placed if st else None,
                        "inputs": st.inputs if st else None,
                        "holds": st.holds if st else None,
                        "top_combo": st.top_combo if st else None,
                        "top_btb": st.top_btb if st else None,
                        "tspins": st.tspins if st else None,
                        "singles": st.singles if st else None,
                        "doubles": st.doubles if st else None,
                        "triples": st.triples if st else None,
                        "quads": st.quads if st else None,
                        "pentas": st.pentas if st else None,
                        "tspin_singles": st.tspin_singles if st else None,
                        "tspin_doubles": st.tspin_doubles if st else None,
                        "tspin_triples": st.tspin_triples if st else None,
                        "tspin_quads": st.tspin_quads if st else None,
                        "all_clears": st.all_clears if st else None,
                        "finesse_combo": st.finesse_combo if st else None,
                        "finesse_faults": st.finesse_faults if st else None,
                        "finesse_perfect_pieces": st.finesse_perfect_pieces if st else None,
                        "garbage_sent": st.garbage_sent if st else None,
                        "garbage_received": st.garbage_received if st else None,
                        "final_time": st.final_time if st else None,
                        "gameover_reason": st.gameover_reason if st else None,
                        "spp": None,  # not stored for replays
                        "time_left": None,
                    },
                })

            partial_q = sess.query(PartialRunRow)
            if gamemode:
                partial_q = partial_q.filter(PartialRunRow.gamemode == gamemode)
            for pr in partial_q.all():
                entries.append({
                    "id": f"partial-{pr.id}",
                    "type": "partial",
                    "gamemode": pr.gamemode,
                    "timestamp": pr.timestamp,
                    "username": None,
                    "stats": {
                        "score": pr.score,
                        "lines": None,
                        "level": None,
                        "apm": None,
                        "pps": pr.pps,
                        "kpp": pr.kpp,
                        "kps": pr.kps,
                        "pieces_placed": pr.pieces_placed,
                        "inputs": pr.inputs,
                        "holds": None,
                        "top_combo": None,
                        "top_btb": None,
                        "tspins": None,
                        "singles": None,
                        "doubles": None,
                        "triples": None,
                        "quads": None,
                        "pentas": None,
                        "tspin_singles": None,
                        "tspin_doubles": None,
                        "tspin_triples": None,
                        "tspin_quads": None,
                        "all_clears": pr.all_clears,
                        "finesse_combo": None,
                        "finesse_faults": None,
                        "finesse_perfect_pieces": None,
                        "garbage_sent": None,
                        "garbage_received": None,
                        "final_time": pr.time_elapsed * 1000 if pr.time_elapsed else None,
                        "gameover_reason": "misdrop",
                        "spp": pr.spp,
                        "time_left": pr.time_left,
                    },
                })

        entries.sort(key=lambda e: e["timestamp"], reverse=True)
        return entries[:limit]

    def get_all_misdrop_data(self, gamemode: str = "blitz",
                             start_date: Optional[datetime] = None,
                             end_date: Optional[datetime] = None) -> tuple[list[dict], list[dict]]:
        """Return (full_replays, partial_runs) for survival analysis.

        If start_date/end_date are provided, only entries within that range are returned.
        """
        full: list[dict] = []
        partial: list[dict] = []
        with self.session() as sess:
            replay_q = (
                sess.query(StatsRow, ReplayRow)
                .join(ReplayRow, ReplayRow.id == StatsRow.replay_id)
                .filter(ReplayRow.gamemode == gamemode)
            )
            if start_date:
                replay_q = replay_q.filter(ReplayRow.timestamp >= start_date)
            if end_date:
                replay_q = replay_q.filter(ReplayRow.timestamp <= end_date)
            replay_q = replay_q.all()

            for st, rp in replay_q:
                full.append({
                    "score": st.score or 0,
                    "pieces": st.pieces_placed or 0,
                    "time": (st.final_time or 0) / 1000,
                    "timestamp": rp.timestamp,
                })

            partial_q = sess.query(PartialRunRow).filter(PartialRunRow.gamemode == gamemode)
            if start_date:
                partial_q = partial_q.filter(PartialRunRow.timestamp >= start_date)
            if end_date:
                partial_q = partial_q.filter(PartialRunRow.timestamp <= end_date)
            for pr in partial_q.all():
                partial.append({
                    "score": pr.score,
                    "pieces": pr.pieces_placed,
                    "time": pr.time_elapsed,
                    "timestamp": pr.timestamp,
                })
        return full, partial
