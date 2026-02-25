# Wolf

LLM Werewolf engine + name bias experiments.

This repo contains two things:

1. **A full Werewolf (Mafia) game engine** where LLM agents play against each other with roles like Seer, Doctor, and Werewolf.
2. **Speed Mafia experiments** — a minimal 1-round variant used to measure name-based bias across LLM models at scale.

## Results

Interactive dashboard with all findings:
**https://huggingface.co/spaces/Queue-Bit-1/llm-bias-dashboard**

## Speed Mafia Experiments

Speed Mafia strips Werewolf down to the minimum needed to detect bias: 7 players, 1 wolf, 1 discussion round, 1 simultaneous vote. 14 API calls per game.

### Setup

```bash
pip install httpx
```

Set your API key(s):

```bash
export OPENAI_API_KEY=sk-...    # for GPT models
export XAI_API_KEY=xai-...      # for Grok models
```

### Running experiments

```bash
# Cloud models (OpenAI, xAI)
python speed_bias_cloud.py 500 --pool large --provider openai --model gpt-4o-mini
python speed_bias_cloud.py 500 --pool large --provider xai --model grok-3-fast

# Local models (requires ollama)
python speed_bias_test.py 200
python speed_bias_test_oss.py 500
```

`--pool` options: `small` (7 names), `medium` (47 names), `large` (140 names across 7 ethnic categories, 70M/70F).

Logs are written to `experiments/logs/`. Each line records one game's speeches, votes, and outcome.

### Generating figures

```bash
python experiments/analysis/generate_figures.py \
  --log experiments/logs/gpt4omini_140n_700g.log \
  --model "GPT-4o-mini" \
  --out experiments/analysis/gpt4omini
```

Produces 7 figures: gender gap, vote flow, top/bottom names, ethnicity breakdown, convergence, wolf immunity, and cross-group targeting. Also exports a JSON summary to `experiments/analysis/results/`.

### Randomization

Each game applies three layers of randomization:
- **Player selection**: `random.sample(pool, 7)` — 7 names drawn from the full pool
- **Seat order**: `random.shuffle(players)` — speaking/voting order randomized
- **Wolf assignment**: `random.choice(players)` — wolf chosen uniformly at random

Speeches and votes are gathered in parallel via `asyncio.gather()` so player order within a round doesn't create sequential bias.

### Data included

| Model | Pool | Games | Log |
|-------|------|------:|-----|
| GPT-4o-mini | 140 | 700 | `experiments/logs/gpt4omini_140n_700g.log` |
| GPT-5-mini | 140 | 2,386 | `experiments/logs/gpt5mini_140n_2386g.log` |
| Grok-3-fast | 140 | 700 | `experiments/logs/grok3fast_140n_700g.log` |
| Grok-4-1-fast | 140 | 700 | `experiments/logs/grok41fast_140n_700g.log` |
| Llama 3.1:8b | 7 | 500 | `experiments/logs/llama_7p_sm_500g.log` |
| gpt-oss:20b | 7 | 500 | `experiments/logs/oss_7p_sm_500g.log` |

## Full Werewolf Engine

A multi-day Werewolf game with night phases, role abilities, and configurable agent types.

### Install

```bash
pip install -e ".[dev]"
```

Requires Python 3.11+.

### Running a game

```bash
# Single game with default config
wolf play

# With options
wolf play --config configs/default_game.yaml --players 7 --model gpt-4o-mini

# Benchmark (multiple games)
wolf benchmark --config configs/pool_benchmark.yaml --games 10 --parallel 2

# With live web GUI
wolf benchmark --config configs/pool_benchmark.yaml --games 5 --gui

# Tournament mode
wolf tournament --config configs/default_game.yaml

# Replay a saved game
wolf replay --game-id results/game_result.json
```

### Roles

- **Werewolf** — kills a villager each night
- **Seer** — inspects one player's role each night
- **Doctor** — protects one player from elimination each night
- **Villager** — no special ability

### Tests

```bash
pytest
```

## Project structure

```
speed_bias_cloud.py          # Cloud experiment script
speed_bias_test.py           # Local model experiment (ollama)
experiments/
  logs/                      # Raw game logs
  analysis/                  # Figures and generate_figures.py
    results/                 # JSON summaries
src/wolf/
  agents/                    # LLM, random, and human agents
  engine/                    # Game state, moderator, resolver
  roles/                     # Role definitions
  session/                   # Game runner, batch runner, tournament
  config/                    # YAML config schema + loader
  metrics/                   # Aggregation and export (JSON, CSV, HTML)
  cli.py                     # Command-line interface
configs/                     # Game configuration files
tests/                       # Unit and integration tests
```

## License

MIT

## Author

Jesse Phillips, 2026
