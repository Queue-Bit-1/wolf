"""Command-line interface for the Wolf benchmark engine."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from typing import Any

import click

logger = logging.getLogger(__name__)


@click.group()
@click.option("--log-level", default="INFO", help="Logging level.")
def cli(log_level: str) -> None:
    """Wolf -- LLM Werewolf Benchmark Engine."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


# ------------------------------------------------------------------
# wolf play
# ------------------------------------------------------------------


@cli.command()
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=False),
    help="Path to game config YAML (default: configs/default_game.yaml).",
)
@click.option("--players", type=int, default=None, help="Override number of players.")
@click.option("--model", type=str, default=None, help="Override default model name.")
@click.option("--verbose", is_flag=True, help="Enable verbose output.")
def play(
    config_path: str | None,
    players: int | None,
    model: str | None,
    verbose: bool,
) -> None:
    """Run a single Werewolf game."""
    from wolf.config.loader import load_config, merge_configs
    from wolf.session.runner import GameRunner

    if verbose:
        logging.getLogger("wolf").setLevel(logging.DEBUG)

    config = load_config(config_path)

    # Apply overrides
    overrides: dict[str, Any] = {}
    if players is not None:
        overrides["num_players"] = players
    if model is not None:
        overrides["default_model"] = {"model": model}
    if overrides:
        config = merge_configs(config, overrides)

    click.echo(
        click.style("=== Wolf: Starting Game ===", fg="cyan", bold=True)
    )
    click.echo(f"  Config: {config.game_name}")
    click.echo(f"  Players: {config.num_players}")
    click.echo(f"  Model: {config.default_model.model}")
    click.echo()

    runner = GameRunner(config)
    result = asyncio.run(runner.run())

    # Print summary
    click.echo()
    click.echo(click.style("=== Game Result ===", fg="green", bold=True))
    click.echo(
        f"  Game ID: {click.style(result.game_id, fg='yellow')}"
    )
    click.echo(
        f"  Winner: {click.style(result.end_event.winning_team, fg='bright_green', bold=True)}"
    )
    click.echo(f"  Reason: {result.end_event.reason}")
    click.echo(f"  Duration: {result.duration:.1f}s")
    click.echo()

    # Player summary
    summary = result.game_summary
    click.echo(click.style("Players:", fg="cyan"))
    for p in summary.get("players", []):
        status = (
            click.style("alive", fg="green")
            if p.get("is_alive")
            else click.style("eliminated", fg="red")
        )
        pid = p.get("player_id", "")
        name = p.get("name", pid)
        role = p.get("role", "?")
        team = p.get("team", "?")
        click.echo(
            f"  {name} ({pid}): {role} [{team}] - {status}"
            f" | speeches={p.get('speeches', 0)}"
            f" | votes={p.get('votes_cast', 0)}"
        )


# ------------------------------------------------------------------
# wolf benchmark
# ------------------------------------------------------------------


@cli.command()
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=False),
    help="Path to game config YAML.",
)
@click.option("--games", type=int, default=10, help="Number of games to run.")
@click.option("--parallel", type=int, default=1, help="Max parallel games.")
@click.option(
    "--output-dir",
    default=None,
    help="Output directory for results (default from config).",
)
@click.option("--gui", is_flag=True, help="Start live web GUI.")
@click.option("--gui-port", type=int, default=8080, help="HTTP port for GUI.")
@click.option("--ws-port", type=int, default=8765, help="WebSocket port for GUI.")
def benchmark(
    config_path: str | None,
    games: int,
    parallel: int,
    output_dir: str | None,
    gui: bool,
    gui_port: int,
    ws_port: int,
) -> None:
    """Run a benchmark suite of multiple games."""
    from wolf.config.loader import load_config
    from wolf.metrics.aggregator import MetricsAggregator
    from wolf.metrics.exporters.csv_exporter import CSVExporter
    from wolf.metrics.exporters.dashboard import DashboardExporter
    from wolf.metrics.exporters.json_exporter import JSONExporter
    from wolf.session.batch import BatchRunner

    config = load_config(config_path)
    out_dir = output_dir or config.metrics.output_dir

    extra_listeners: list[Any] = []
    if gui:
        from wolf.web import WebEventListener, start_web_server

        web_listener = WebEventListener()
        extra_listeners.append(web_listener)

    click.echo(
        click.style("=== Wolf: Benchmark Suite ===", fg="cyan", bold=True)
    )
    click.echo(f"  Games: {games}")
    click.echo(f"  Parallel: {parallel}")
    click.echo(f"  Output: {out_dir}")
    if gui:
        click.echo(f"  Web GUI: http://localhost:{gui_port}")
    click.echo()

    batch = BatchRunner(config, extra_listeners=extra_listeners)

    if gui:
        async def _run_with_gui() -> list[Any]:
            task = asyncio.create_task(
                start_web_server(web_listener, http_port=gui_port, ws_port=ws_port)
            )
            results = await batch.run(num_games=games, parallel=parallel)
            task.cancel()
            return results

        results = asyncio.run(_run_with_gui())
    else:
        results = asyncio.run(batch.run(num_games=games, parallel=parallel))

    # Aggregate metrics
    summaries = [r.game_summary for r in results]
    aggregator = MetricsAggregator()
    aggregated = aggregator.aggregate(summaries)

    # Export results
    os.makedirs(out_dir, exist_ok=True)

    formats = config.metrics.export_formats

    if "json" in formats:
        exporter = JSONExporter()
        exporter.export(aggregated, os.path.join(out_dir, "benchmark_results.json"))
        click.echo(f"  Exported JSON to {out_dir}/benchmark_results.json")

    if "csv" in formats:
        exporter_csv = CSVExporter()
        exporter_csv.export(aggregated, os.path.join(out_dir, "benchmark_results.csv"))
        click.echo(f"  Exported CSV to {out_dir}/benchmark_results.csv")

    if "html" in formats:
        exporter_html = DashboardExporter()
        exporter_html.export(
            aggregated, os.path.join(out_dir, "benchmark_dashboard.html")
        )
        click.echo(f"  Exported HTML to {out_dir}/benchmark_dashboard.html")

    # Print comprehensive stats
    click.echo()
    click.echo(click.style("=" * 60, bold=True))
    click.echo(click.style("  BENCHMARK RESULTS", bold=True))
    click.echo(click.style("=" * 60, bold=True))
    click.echo(f"  Games completed: {len(results)} / {games}")

    # Durations
    durations = [r.duration for r in results]
    avg_dur = sum(durations) / len(durations) if durations else 0
    click.echo(f"  Avg duration: {avg_dur:.0f}s  Total: {sum(durations):.0f}s")

    # Team win rates
    click.echo()
    click.echo(click.style("  TEAM WIN RATES", bold=True))
    win_rates = aggregated.get("win_rates", {})
    team_rates = win_rates.get("win_rate_by_team", {})
    for team, rate in sorted(team_rates.items()):
        bar = "#" * int(rate * 20)
        click.echo(f"    {team:<12} {rate*100:5.1f}%  {bar}")

    # Role stats
    click.echo()
    click.echo(click.style("  ROLE STATS", bold=True))
    click.echo(f"    {'Role':<12} {'Win%':>6} {'Surv%':>6} {'AvgDays':>8}")
    click.echo(f"    {'─'*34}")
    role_wr = win_rates.get("win_rate_by_role", {})
    role_sr = win_rates.get("survival_rate_by_role", {})
    role_st = win_rates.get("avg_survival_time_by_role", {})
    for role in sorted(set(list(role_wr) + list(role_sr))):
        wr = role_wr.get(role, 0)
        sr = role_sr.get(role, 0)
        st = role_st.get(role, 0)
        click.echo(f"    {role:<12} {wr*100:5.1f}% {sr*100:5.1f}% {st:7.1f}")

    # Model comparison
    click.echo()
    click.echo(click.style("  MODEL COMPARISON", bold=True))
    model_comp = aggregated.get("model_comparison", {})
    model_wr = win_rates.get("win_rate_by_model", {})
    click.echo(f"    {'Model':<22} {'Win%':>6} {'Surv%':>6} {'AvgSpch':>8} {'N':>4}")
    click.echo(f"    {'─'*48}")
    for model in sorted(model_comp.keys()):
        stats = model_comp[model]
        wr = model_wr.get(model, 0)
        sv = stats.get("survival", {}).get("mean", 0)
        sp = stats.get("speeches", {}).get("mean", 0)
        n = stats.get("wins", {}).get("n", 0)
        click.echo(f"    {model:<22} {wr*100:5.1f}% {sv*100:5.1f}% {sp:7.1f} {n:>4}")

    # Cross-game stats
    cross_game = aggregated.get("cross_game", {})
    game_length = cross_game.get("game_length", {})
    if game_length:
        click.echo()
        click.echo(
            f"  Avg game length: {game_length.get('mean', 0):.1f} days"
            f" (std: {game_length.get('std', 0):.2f},"
            f" 95%CI: [{game_length.get('ci_lower', 0):.1f}-{game_length.get('ci_upper', 0):.1f}])"
        )

    # Per-game results
    click.echo()
    click.echo(click.style("  PER-GAME RESULTS", bold=True))
    click.echo(f"    {'#':<4} {'Winner':<12} {'Days':>5} {'Duration':>10}")
    click.echo(f"    {'─'*33}")
    for i, r in enumerate(results, 1):
        winner = r.end_event.winning_team
        days = r.game_summary.get("total_days", 0)
        click.echo(f"    {i:<4} {winner:<12} {days:>5} {r.duration:>9.0f}s")

    click.echo(click.style("=" * 60, bold=True))


# ------------------------------------------------------------------
# wolf tournament
# ------------------------------------------------------------------


@cli.command()
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=False),
    help="Path to tournament config YAML.",
)
@click.option(
    "--output-dir",
    default=None,
    help="Output directory for results.",
)
def tournament(config_path: str | None, output_dir: str | None) -> None:
    """Run a round-robin tournament across model configurations."""
    from wolf.config.loader import load_config
    from wolf.metrics.exporters.csv_exporter import CSVExporter
    from wolf.metrics.exporters.dashboard import DashboardExporter
    from wolf.metrics.exporters.json_exporter import JSONExporter
    from wolf.session.tournament import TournamentRunner

    config = load_config(config_path)
    out_dir = output_dir or config.metrics.output_dir

    click.echo(
        click.style("=== Wolf: Tournament ===", fg="cyan", bold=True)
    )

    # For a single-config tournament, we just run that config
    # A multi-config tournament would load multiple configs from a
    # tournament YAML; for now we support single-config mode.
    runner = TournamentRunner(configs=[config])
    result = asyncio.run(runner.run())

    # Export
    os.makedirs(out_dir, exist_ok=True)

    json_exp = JSONExporter()
    json_exp.export(
        result.aggregated_metrics,
        os.path.join(out_dir, "tournament_results.json"),
    )

    csv_exp = CSVExporter()
    csv_exp.export(
        result.aggregated_metrics,
        os.path.join(out_dir, "tournament_results.csv"),
    )

    dash_exp = DashboardExporter()
    dash_exp.export(
        result.aggregated_metrics,
        os.path.join(out_dir, "tournament_dashboard.html"),
    )

    click.echo()
    click.echo(
        click.style("=== Tournament Results ===", fg="green", bold=True)
    )
    click.echo(f"  Total games: {len(result.results)}")
    click.echo(f"  Output: {out_dir}")

    # Print rankings
    if result.model_rankings:
        click.echo()
        click.echo(click.style("Model Rankings:", fg="cyan"))
        for entry in result.model_rankings:
            rank = entry.get("rank", "?")
            model = entry.get("model", "unknown")
            win_rate = entry.get("win_rate", 0)
            composite = entry.get("composite_score", 0)
            click.echo(
                f"  #{rank} {click.style(model, fg='yellow')}"
                f" | win_rate={win_rate:.3f}"
                f" | composite={composite:.3f}"
            )


# ------------------------------------------------------------------
# wolf replay
# ------------------------------------------------------------------


@cli.command()
@click.option(
    "--game-id",
    required=True,
    type=click.Path(exists=True),
    help="Path to a JSON game result file.",
)
def replay(game_id: str) -> None:
    """Replay a game from a JSON result file."""
    with open(game_id, "r", encoding="utf-8") as f:
        data = json.load(f)

    click.echo(
        click.style("=== Wolf: Game Replay ===", fg="cyan", bold=True)
    )

    # Print game metadata
    if "game_id" in data:
        click.echo(f"  Game ID: {data['game_id']}")
    if "duration" in data:
        click.echo(f"  Duration: {data['duration']:.1f}s")

    # Print result
    result = data.get("result", data.get("game_summary", {}).get("result", {}))
    if result:
        click.echo()
        click.echo(
            click.style(
                f"  Winner: {result.get('winning_team', '?')}",
                fg="bright_green",
                bold=True,
            )
        )
        click.echo(f"  Reason: {result.get('reason', 'N/A')}")

    # Print players
    players = data.get("players", data.get("game_summary", {}).get("players", []))
    if players:
        click.echo()
        click.echo(click.style("Players:", fg="cyan"))
        for p in players:
            pid = p.get("player_id", "")
            name = p.get("name", pid)
            role = p.get("role", "?")
            team = p.get("team", "?")
            alive = p.get("is_alive", False)
            status_str = (
                click.style("alive", fg="green")
                if alive
                else click.style(f"eliminated (day {p.get('survived_until', '?')})", fg="red")
            )
            click.echo(f"  {name} ({pid}): {role} [{team}] - {status_str}")

    # Print event-by-event replay
    _replay_events(data)


def _replay_events(data: dict[str, Any]) -> None:
    """Print a simplified event-by-event replay from game data."""
    players = data.get("players", data.get("game_summary", {}).get("players", []))
    if not players:
        return

    click.echo()
    click.echo(click.style("=== Event Timeline ===", fg="cyan", bold=True))

    total_days = data.get("total_days", data.get("game_summary", {}).get("total_days", 0))

    # Build a name lookup
    name_map: dict[str, str] = {}
    for p in players:
        name_map[p.get("player_id", "")] = p.get("name", p.get("player_id", ""))

    # Reconstruct events from player data
    for p in players:
        pid = p.get("player_id", "")
        name = name_map.get(pid, pid)

        # Speeches
        for i, speech in enumerate(p.get("speech_contents", [])):
            excerpt = speech[:120] + "..." if len(speech) > 120 else speech
            click.echo(
                f"  {click.style(f'[Speech]', fg='blue')} "
                f"{click.style(name, fg='yellow')}: {excerpt}"
            )

        # Votes
        for target in p.get("vote_targets", []):
            if target is not None:
                target_name = name_map.get(target, target)
                click.echo(
                    f"  {click.style(f'[Vote]', fg='magenta')} "
                    f"{click.style(name, fg='yellow')} -> {target_name}"
                )

        # Elimination
        if not p.get("is_alive", True):
            cause = p.get("elimination_cause", "unknown")
            survived = p.get("survived_until", "?")
            click.echo(
                f"  {click.style(f'[Eliminated]', fg='red')} "
                f"{click.style(name, fg='yellow')} on day {survived} ({cause})"
            )
