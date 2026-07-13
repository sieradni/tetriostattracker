import os
from pathlib import Path

import click

from ttr_tracker.database import Database
from ttr_tracker.importer import import_directory, import_file
from ttr_tracker.queries import (
    get_replay_detail,
    get_session_summaries,
    get_summary,
    get_trends,
)

DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "tracker.db"
DEFAULT_REPLAY_DIR = Path(__file__).resolve().parent.parent / "data" / "replays"


@click.group()
@click.option("--db", default=str(DEFAULT_DB), show_default=True, help="Database path")
@click.pass_context
def cli(ctx: click.Context, db: str) -> None:
    ctx.ensure_object(dict)
    ctx.obj["db"] = Database(db)


@cli.group()
def partial():
    """Manage partial runs (misdrop screenshots)."""


@partial.command(name="add")
@click.option("--timestamp", required=True, help="ISO timestamp")
@click.option("--score", type=int, required=True)
@click.option("--pieces", type=int, required=True)
@click.option("--pps", type=float, required=True)
@click.option("--inputs", type=int, required=True)
@click.option("--kpp", type=float, required=True)
@click.option("--spp", type=float, required=True)
@click.option("--all-clears", type=int, required=True)
@click.option("--time-left", type=float, required=True, help="Seconds remaining")
@click.option("--notes", default=None)
@click.pass_context
def add_partial(ctx, timestamp, score, pieces, pps, inputs, kpp, spp, all_clears, time_left, notes):
    db: Database = ctx.obj["db"]
    from datetime import datetime
    ts = datetime.fromisoformat(timestamp)
    objective_ms = 120000
    time_elapsed = max(0, objective_ms - time_left * 1000) / 1000
    kps = inputs / time_elapsed if time_elapsed > 0 else 0.0
    rid = db.insert_partial_run(
        timestamp=ts, gamemode="blitz",
        score=score, pieces_placed=pieces, pps=pps, inputs=inputs,
        kpp=kpp, spp=spp, all_clears=all_clears,
        time_left=time_left, time_elapsed=time_elapsed, kps=kps,
        notes=notes,
    )
    click.echo(f"Added partial run #{rid}")


@partial.command(name="list")
@click.option("--limit", default=50)
@click.pass_context
def list_partial(ctx, limit):
    db: Database = ctx.obj["db"]
    rows = db.get_all_partial_runs(limit=limit)
    if not rows:
        click.echo("No partial runs.")
        return
    click.echo(f"{'ID':<5} {'Date':<22} {'Score':<10} {'Pieces':<8} {'PPS':<8} {'KPP':<8} {'KPS':<8} {'Time':<8}")
    click.echo("-" * 85)
    for r in rows:
        ts = r.timestamp.strftime("%Y-%m-%d %H:%M")
        click.echo(f"{r.id:<5} {ts:<22} {r.score:<10} {r.pieces_placed:<8} {r.pps:<8.2f} {r.kpp:<8.2f} {r.kps:<8.2f} {r.time_elapsed:<8.1f}")


@partial.command(name="delete")
@click.argument("run_id", type=int)
@click.pass_context
def delete_partial(ctx, run_id):
    db: Database = ctx.obj["db"]
    if db.delete_partial_run(run_id):
        click.echo(f"Deleted partial run #{run_id}")
    else:
        click.echo(f"Partial run #{run_id} not found")


@cli.command()
@click.argument("path", type=click.Path(exists=True), default=str(DEFAULT_REPLAY_DIR))
@click.pass_context
def import_replays(ctx: click.Context, path: str) -> None:
    db: Database = ctx.obj["db"]
    p = Path(path)
    if p.is_dir():
        results = import_directory(db, p)
    else:
        results = [import_file(db, p)]
    for msg in results:
        click.echo(msg)
    click.echo(f"Total replays in DB: {db.count_replays()}")


@cli.command()
@click.option("--gamemode", default="blitz", help="Filter by gamemode")
@click.option("--limit", default=30, help="Number of replays to show")
@click.pass_context
def list_replays(ctx: click.Context, gamemode: str, limit: int) -> None:
    db: Database = ctx.obj["db"]
    rows = db.get_all_replays(gamemode=gamemode, limit=limit)
    if not rows:
        click.echo("No replays found.")
        return

    click.echo(f"{'ID':<20} {'Date':<22} {'Score':<10} {'Lines':<6} {'APM':<8} {'PPS':<8} {'KPP':<8} {'KPS':<8} {'Level':<6}")
    click.echo("-" * 100)
    for replay, stats in rows:
        score = stats.score if stats else 0
        lines = stats.lines if stats else 0
        apm = f"{stats.apm:.2f}" if stats and stats.apm else "0"
        pps = f"{stats.pps:.2f}" if stats and stats.pps else "0"
        kpp = f"{stats.kpp:.2f}" if stats and stats.kpp else "0"
        kps = f"{stats.kps:.2f}" if stats and stats.kps else "0"
        level = stats.level if stats else 1
        ts = replay.timestamp.strftime("%Y-%m-%d %H:%M")
        click.echo(f"{replay.id:<20} {ts:<22} {score:<10} {lines:<6} {apm:<8} {pps:<8} {kpp:<8} {kps:<8} {level:<6}")


@cli.command()
@click.option("--limit", default=30, help="Number of matches to show")
@click.pass_context
def list_matches(ctx: click.Context, limit: int) -> None:
    db: Database = ctx.obj["db"]
    matches = db.get_all_matches(limit=limit)
    if not matches:
        click.echo("No matches found.")
        return
    click.echo(f"{'ID':<20} {'Date':<22} {'Player 1':<14} {'W1':<4} {'Player 2':<14} {'W2':<4} {'Winner':<14}")
    click.echo("-" * 100)
    for m in matches:
        ts = m.timestamp.strftime("%Y-%m-%d %H:%M")
        click.echo(f"{m.id:<20} {ts:<22} {m.player1_name:<14} {m.player1_wins:<4} {m.player2_name:<14} {m.player2_wins:<4} {m.winner_name:<14}")


@cli.command()
@click.option("--gamemode", default="blitz")
@click.pass_context
def summary(ctx: click.Context, gamemode: str) -> None:
    db: Database = ctx.obj["db"]
    s = get_summary(db, gamemode)
    if s["total"] == 0:
        click.echo(f"No replays for gamemode '{gamemode}'.")
        return

    click.echo(f"Summary for '{gamemode}':")
    click.echo(f"  Total replays: {s['total']}")
    click.echo(f"  Sessions:      {s['session_count']}")
    click.echo()
    if s["recent_avg"]:
        click.echo("  Last 20 game averages:")
        for k, v in s["recent_avg"].items():
            click.echo(f"    {k}: {v:.1f}")
    click.echo()
    if s["pbs"]:
        click.echo("  Personal Bests:")
        for k, (v, rid, ts) in sorted(s["pbs"].items(), key=lambda x: x[1][0], reverse=True):
            click.echo(f"    {k}: {v:.1f}")


@cli.command()
@click.option("--port", default=8080, help="Port to serve on")
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.pass_context
def serve(ctx: click.Context, port: int, host: str) -> None:
    db: Database = ctx.obj["db"]
    db_path = ctx.obj["db"].engine.url.database
    click.echo(f"Starting dashboard on http://{host}:{port}")
    click.echo(f"Database: {db_path}")

    os.environ["TTR_DB_PATH"] = db_path
    import uvicorn
    uvicorn.run("ttr_tracker.dashboard.app:app", host=host, port=port, reload=False)


@cli.command()
@click.option("--gamemode", default="blitz", help="Filter by gamemode")
@click.argument("output", type=click.Path(), default="export.csv")
@click.pass_context
def export(ctx: click.Context, gamemode: str, output: str) -> None:
    db: Database = ctx.obj["db"]
    entries = db.get_combined_entries(gamemode=gamemode, limit=10000)
    if not entries:
        click.echo("No data to export.")
        return

    import csv

    fields = [
        "score", "lines", "level", "apm", "pps", "kpp", "kps",
        "pieces_placed", "inputs", "holds", "top_combo", "top_btb", "tspins",
        "singles", "doubles", "triples", "quads", "pentas",
        "all_clears", "finesse_faults", "finesse_perfect_pieces",
        "final_time", "gameover_reason",
    ]

    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "type", "timestamp", "username", "gamemode"] + fields)
        for e in entries:
            s = e["stats"]
            row_vals = []
            for k in fields:
                v = s.get(k)
                row_vals.append(v if v is not None else "")
            writer.writerow([
                e["id"], e["type"], e["timestamp"].isoformat() if hasattr(e["timestamp"], "isoformat") else e["timestamp"],
                e.get("username") or "", e["gamemode"],
            ] + row_vals)
    click.echo(f"Exported {len(entries)} entries to {output}")


if __name__ == "__main__":
    cli()
