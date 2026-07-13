from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
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
    created_at = Column(DateTime, default=datetime.utcnow)

    stats = relationship("StatsRow", back_populates="replay", uselist=False)
    options = relationship("OptionsRow", back_populates="replay", uselist=False)


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
    created_at = Column(DateTime, default=datetime.utcnow)

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
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        with Session(self.engine) as sess:
            yield sess

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
                timestamp=timestamp,
                username=username,
                user_id=user_id,
                version=version,
            )
            sess.add(row)
            sess.commit()

    def insert_stats(
        self,
        replay_id: str,
        gamemode: str,
        **kwargs,
    ) -> None:
        with self.session() as sess:
            row = StatsRow(replay_id=replay_id, gamemode=gamemode, **kwargs)
            sess.add(row)
            sess.commit()

    def insert_options(
        self,
        replay_id: str,
        **kwargs,
    ) -> None:
        with self.session() as sess:
            row = OptionsRow(replay_id=replay_id, **kwargs)
            sess.add(row)
            sess.commit()

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

    def get_pbs(self, gamemode: str) -> dict[str, tuple[float, str, datetime]]:
        pb_columns = [
            "score", "lines", "apm", "pps", "level", "top_combo", "top_btb",
            "tspins", "quads", "all_clears", "finesse_perfect_pieces",
            "kpp", "kps", "final_time",
        ]
        result = {}
        with self.session() as sess:
            for col_name in pb_columns:
                col = getattr(StatsRow, col_name, None)
                if col is None:
                    continue
                order = col.asc() if col_name == "final_time" else col.desc()
                row = (
                    sess.query(col, ReplayRow.id, ReplayRow.timestamp)
                    .join(ReplayRow, ReplayRow.id == StatsRow.replay_id)
                    .filter(ReplayRow.gamemode == gamemode)
                    .order_by(order)
                    .first()
                )
                if row and row[0] is not None:
                    result[col_name] = (float(row[0]), row[1], row[2])
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
                timestamp=timestamp,
                player1_id=player1_id,
                player1_name=player1_name,
                player2_id=player2_id,
                player2_name=player2_name,
                player1_wins=player1_wins,
                player2_wins=player2_wins,
                winner_name=winner_name,
            )
            sess.add(row)
            sess.commit()

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
        self, player_id: str, limit: int = 200
    ) -> list[dict]:
        with self.session() as sess:
            matches = (
                sess.query(MatchRow)
                .filter(
                    (MatchRow.player1_id == player_id) | (MatchRow.player2_id == player_id)
                )
                .order_by(MatchRow.timestamp.asc())
                .limit(limit)
                .all()
            )
            result = []
            for m in matches:
                rounds = (
                    sess.query(MatchRoundRow)
                    .filter_by(match_id=m.id, player_id=player_id)
                    .all()
                )
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

    def get_session_groups(self, gamemode: str, gap_minutes: int = 60) -> list[list[ReplayRow]]:
        with self.session() as sess:
            rows = (
                sess.query(ReplayRow)
                .filter(ReplayRow.gamemode == gamemode)
                .order_by(ReplayRow.timestamp.asc())
                .all()
            )
        if not rows:
            return []
        groups: list[list[ReplayRow]] = [[rows[0]]]
        for r in rows[1:]:
            gap = (r.timestamp - groups[-1][-1].timestamp).total_seconds() / 60
            if gap > gap_minutes:
                groups.append([])
            groups[-1].append(r)
        return groups
