# 🎯 LLM Strategy Learner V2: Opponent Archetype Discovery & Championship Tournament

An experiment exploring how a Large Language Model (**OpenRouter**, **Google Gemini**, or **Local Ollama**) autonomously identifies opponent decision rules in the **Iterated Prisoner's Dilemma (IPD)**, synthesizes strategic meta-knowledge through a **Dreamer Phase**, and competes in an **All-Play-All Round-Robin Championship Tournament**.

---

## ⚡ Key Upgrades in V2

1. **Zero Environmental Noise (Deterministic IPD)**:
   - Eliminated the 5% move-flip error rate from V1 to enable predictable, reproducible game-theoretic deduction and eliminate accidental retaliation spirals.
2. **Dynamic Strategy Identification Prompts (4 Configurable Modes)**:
   - **Mode 1 (`explicit_strategies`)**: Agent is given the exact candidate strategy catalog and tasked with probing early to recognize and counter them.
   - **Mode 2 (`generic_archetypes`)**: Agent knows classical heuristics exist (reciprocal, unforgiving, exploitative, probing) and formulates hypotheses to deduce the decision rule.
   - **Mode 3 (`blind_optimization`)**: Tabula rasa — agent is given only IPD rules and payoff matrix, optimizing purely through empirical observation.
   - **Mode 4 (`revealed_strategy`)**: Agent is told the exact strategy identity it is facing and can focus on the optimal counter-strategy.
   - Every mode is told that matches are `Normal(10, 1)` rounds long, without revealing the exact sampled length.
3. **Structured Sequential Block Training**:
   - Replaces random opponent selection with 5 consecutive matches per strategy archetype (*Always Cooperate, Always Defect, Tit For Tat, Suspicious TFT, Tit For Two Tats, Tat For Tit, Random, Detective*), allowing deep probing and reflection.
4. **🌙 The Dreamer Phase (Memory Consolidation)**:
   - Synthesizes all raw training notes, reflections, and match outcomes into a single high-density **Master Strategy Playbook** (`output/master_strategy_playbook.md`) that is frozen into the LLM's long-term memory for evaluation.
5. **🏆 Round-Robin Championship Ranking System**:
   - Replaces evolutionary spatial networks with a chess-championship ranking system: every strategy plays every other strategy 5 times.
   - Tracks Chess Match Points ($W=1.0, D=0.5, L=0.0$), total IPD score, and produces an interactive **Head-to-Head Cross-Table Matrix** and HTML dashboard.
6. **Unified Master Configuration (`config.toml`)**:
   - Every parameter (prompts, full system prompt templates, models, error rates, matches, rounds, scoring points, participants) is centrally configurable from `config.toml`, complete with comments and documentation.

---

## 📂 Repository Architecture

```
llm-strategy-learner/
├── config.toml                 # Master configuration file (TOML with comments & system prompts)
├── main.py                     # Main CLI entry point
├── requirements.txt            # Project dependencies
├── README.md                   # V2 Documentation
├── src/
│   ├── config.py               # Pydantic schema and config loader
│   ├── constants.py            # Move enum and deterministic payoff matrix
│   ├── game_machine.py         # Pure deterministic IPD game engine
│   ├── player.py               # Base Player class
│   ├── player_classes.py       # Classical strategy catalog & registry
│   ├── prompts.py              # Modular prompt generator (Modes 1, 2, 3, Reflection, Dreamer)
│   ├── llm_player.py           # Structured output LLM player (OpenRouter, Gemini, Ollama)
│   ├── dreamer.py              # Dreamer memory synthesis engine
│   ├── training_battery.py     # Sequential 5-match strategy block trainer
│   ├── tournament.py           # Round-Robin Championship tournament engine
│   ├── usage_tracker.py        # API rate-limit and quota tracker
│   └── reporting/
│       ├── __init__.py
│       └── tournament_report.py # Rich TUI tables, Markdown leaderboard, and HTML dashboard
├── tests/                      # Automated test suite
│   ├── test_config.py
│   ├── test_game_machine.py
│   ├── test_player_classes.py
│   └── test_tournament.py
└── output/                     # Generated logs, playbooks, leaderboards, HTML reports
```

---

## 🚀 Quickstart Guide

### 1. Installation

```powershell
# Activate your virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment & API Keys

Create or edit `.env` in the root directory:

```env
# OpenRouter (Default)
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_MODEL=stealth/ox-alpha

# Google Gemini Direct (Alternative)
GEMINI_API_KEY=your_gemini_key
GEMINI_MODEL=gemini-2.5-flash

# Local Ollama (Alternative; start Ollama before running)
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

### 3. Run the Simulation

#### A. Full Run (Training -> Dreamer -> Championship Tournament):
```powershell
python main.py
```

To use Ollama for a run, set the provider and model in `.env`:

```powershell
python main.py --provider ollama
```

You can override either value for one run without editing `.env`:

```powershell
python main.py --provider ollama --model qwen2.5:7b
```

Ollama must be running locally and the selected model must already be available
(`ollama pull qwen2.5:7b`).

#### B. Quick Classical Dry Run (Zero LLM API cost):
Runs the championship tournament exclusively between classical strategies to verify scoring and reporting instantly:
```powershell
python main.py --no-llm
```

#### C. Change Prompt Modes on the Fly:
```powershell
# Run with Mode 1: Known Strategy Pool
python main.py --mode explicit_strategies

# Run with Mode 2: Classical Heuristic Archetypes
python main.py --mode generic_archetypes

# Run with Mode 3: Tabula Rasa / Blind
python main.py --mode blind_optimization
```

#### D. Training-Only or Evaluation-Only:
```powershell
# Only run block training and Dreamer synthesis
python main.py --train-only

# Run tournament using pre-computed master playbook
python main.py --eval-only
```

---

## 🛠️ Configuration (`config.toml`)

All runtime options, strategy pools, and full system prompts are customizable in `config.toml`:

```toml
[experiment]
name = "v2_championship_tournament"
output_dir = "output"

[llm]
provider = "openrouter"
model_name = "stealth/ox-alpha"
temperature = 0.2

[environment]
error_rate = 0.0 # Deterministic (zero noise)

[prompts]
# Mode 1: explicit_strategies | Mode 2: generic_archetypes | Mode 3: blind_optimization
prompt_mode = "explicit_strategies"

[prompts.system_prompts]
# Fully customizable system prompts with multi-line text directly in TOML
explicit_strategies = """You are an expert game theory agent playing the Iterated Prisoner's Dilemma..."""

[training]
matches_per_strategy = 5
rounds_per_match = { average = 10.0, std_dev = 1.0 }
strategies = [
    "AlwaysCooperate", "AlwaysDefect", "TitForTat",
    "SuspiciousTitForTat", "TitForTwoTats", "TatForTit",
    "Random", "Detective"
]

[evaluation]
matches_per_pairing = 5
rounds_per_match = { average = 10.0, std_dev = 1.0 }
```

---

## 📊 Outputs & Reports

After running, check the `output/` directory:
- `output/master_strategy_playbook.md`: Synthesized strategic playbook produced by the Dreamer phase.
- `output/training_history.json`: Turn-by-turn and match reflection logs across all training strategy blocks.
- `output/tournament_leaderboard.md`: Markdown standings table with W-D-L records and cross-table matrix.
- `output/tournament_leaderboard.json`: Raw tournament statistics and match results.
- `output/tournament_report.html`: Beautiful interactive dashboard featuring metric cards, leaderboard, and color-coded head-to-head cross-table.
