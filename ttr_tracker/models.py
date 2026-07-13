from __future__ import annotations
from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field


class User(BaseModel):
    id: str
    username: str
    avatar_revision: Optional[int] = None
    flags: Optional[int] = None
    country: Optional[str] = None


class KeyEvent(BaseModel):
    frame: int
    type: str
    data: dict[str, Any]


class Handling(BaseModel):
    arr: int = 0
    das: int = 4
    dcd: int = 0
    sdf: int = 41
    safelock: bool = True
    cancel: bool = True
    may20g: bool = True
    irs: str = "tap"
    ihs: str = "tap"


class ReplayOptions(BaseModel):
    version: int = 19
    anchorseed: bool = True
    allow180: bool = True
    objective_type: str = "timed"
    objective_time: int | None = 120000
    levels: bool | None = True
    levelspeed: float | None = 0.42
    levelgbase: float | None = 0.65
    gravitymay20g: bool | None = False
    slot_bar2: str = "progress"
    can_retry: bool = True
    nolockout: bool = True
    gameid: int = 2
    seed: int = 0
    handling: Handling = Field(default_factory=Handling)
    countdown: bool = True
    precountdown: int = 5000
    prestart: int = 1000
    mission: str = ""
    zoominto: str = "slow"
    slot_counter1: str = "timer"
    slot_counter2: str = "lines"
    slot_counter3: str = "level"
    slot_counter4: str = "finesse_l"
    slot_counter5: str = "score"
    pro: bool = True
    stride: bool = True
    no_szo: bool = True
    seed_random: bool = False


class ClearStats(BaseModel):
    singles: int = 0
    doubles: int = 0
    triples: int = 0
    quads: int = 0
    pentas: int = 0
    realtspins: int = 0
    minitspins: int = 0
    minitspinsingles: int = 0
    tspinsingles: int = 0
    minitspindoubles: int = 0
    tspindoubles: int = 0
    minitspintriples: int = 0
    tspintriples: int = 0
    minitspinquads: int = 0
    tspinquads: int = 0
    tspinpentas: int = 0
    allclear: int = 0


class GarbageStats(BaseModel):
    sent: int = 0
    sent_nomult: int = 0
    maxspike: int = 0
    maxspike_nomult: int = 0
    received: int = 0
    attack: int = 0
    cleared: int = 0


class FinesseStats(BaseModel):
    combo: int = 0
    faults: int = 0
    perfectpieces: int = 0


class ZenithStats(BaseModel):
    altitude: int = 0
    rank: int = 1
    peakrank: int = 1
    avgrankpts: float = 0
    floor: int = 0
    targetingfactor: int = 3
    targetinggrace: int = 0
    totalbonus: int = 0
    revives: int = 0
    revivesTotal: int = 0
    revivesMaxOfBoth: int = 0
    speedrun: bool = False
    speedrun_seen: bool = False
    splits: list[int] = Field(default_factory=lambda: [0] * 9)


class AggregateStats(BaseModel):
    apm: float = 0.0
    pps: float = 0.0
    vsscore: float = 0.0


class ResultsStats(BaseModel):
    lines: int = 0
    level_lines: int = 0
    level_lines_needed: int = 15
    inputs: int = 0
    holds: int = 0
    score: int = 0
    zenlevel: int = 1
    zenprogress: int = 0
    level: int = 1
    combo: int = 0
    topcombo: int = 0
    combopower: int = 0
    btb: int = 1
    topbtb: int = 1
    btbpower: int = 0
    tspins: int = 0
    piecesplaced: int = 0
    clears: ClearStats = Field(default_factory=ClearStats)
    garbage: GarbageStats = Field(default_factory=GarbageStats)
    kills: int = 0
    finesse: FinesseStats = Field(default_factory=FinesseStats)
    zenith: ZenithStats = Field(default_factory=ZenithStats)
    finaltime: float = 0.0


class ReplayResults(BaseModel):
    aggregatestats: AggregateStats = Field(default_factory=AggregateStats)
    stats: ResultsStats = Field(default_factory=ResultsStats)
    gameoverreason: str = ""


class Replay(BaseModel):
    frames: int = 0
    events: list[KeyEvent] = Field(default_factory=list)
    options: ReplayOptions = Field(default_factory=ReplayOptions)
    results: ReplayResults = Field(default_factory=ReplayResults)


class TTRFile(BaseModel):
    id: str
    gamemode: str
    ts: datetime
    users: list[User] = Field(default_factory=list)
    replay: Replay = Field(default_factory=Replay)
    version: int = 1
