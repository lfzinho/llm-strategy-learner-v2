# LLM Strategy Learner V2

An experiment on whether a language model can **work out an opponent's rule from its moves, remember what it learned, and turn that into points** in a repeated two-player game (the iterated Prisoner's Dilemma, shown to the model with neutral labels *Alpha* = cooperate and *Beta* = defect).

The pipeline has three stages:

1. **Training**: the model plays blocks of matches against classical strategies. After each match it reflects and stores a lesson in a memory stream.
2. **Dreamer**: the memory plus the *measured* training results are consolidated into a playbook.
3. **Tournament**: the model, now frozen, plays a round-robin against every strategy.

What you can do with it: compare how much information the prompt gives the model (five **tiers**), see in a **scorecard** whether the model played well and why not, and **edit every prompt in one file**.

---

## Quickstart

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`.env` (keys are read from the environment; `mock` needs none):

```env
LLM_PROVIDER=openrouter            # openrouter | gemini | ollama | mock
OPENROUTER_API_KEY=...
OPENROUTER_MODEL=...
GEMINI_API_KEY=...   GEMINI_MODEL=gemini-2.5-flash
OLLAMA_HOST=http://localhost:11434   OLLAMA_MODEL=qwen2.5:3b
```

```powershell
python main.py --provider mock --quick        # offline smoke test of the whole pipeline (seconds, free)
python main.py --show-prompt t1_informed      # print the exact prompt a tier produces (no LLM call)
python main.py --tier t4_blind                # full run: training -> Dreamer -> tournament
python main.py --tiers all --quick            # run every tier and print a comparison table
python main.py --no-llm                       # classical strategies only
python main.py --train-only                   # or --eval-only to reuse the saved playbook and memory
```

If a run stops (API outage, quota), progress is checkpointed. Run the same command again to resume.

---

## The five prompt tiers

Tiers form an information ladder, from "told everything" to "figure it out". They are defined in [`prompts.toml`](prompts.toml) as ordered lists of reusable blocks, derived from the sketch in `docs/LLM Instructions.md`.

| Tier | Opponent | Counter-strategy | Match length | Notes |
|---|---|---|---|---|
| `t1_informed` | named + described | **given** | exact (counts down) | plus an end-game hint |
| `t2_named` | named + described | no | exact | |
| `t3_pool_hint` | hidden | no | about N rounds | full list of possible strategies shown |
| `t4_blind` | hidden | no | about N rounds | no list, infer from behaviour |
| `t5_open_ended` | hidden | no | unknown (random, may be long) | use `rounds_per_match = { prob_end = 0.1 }` |

Opponents are given neutral names in prompts (for example Tit For Tat is *Alpha Mimic*, Grim Trigger is *Beta Resentful*) so the model cannot lean on what it remembers about the textbook names. Tiers `t1` and `t2` also name the opponent in reflections; the hidden tiers never do, and the Dreamer labels opponents "Opponent type 1, 2, ..." there so the playbook cannot leak the pool.

The counters in `t1` are **verified**: `tests/test_opponents.py` plays each counter against the real Axelrod strategy and requires it to score within 10% of the brute-force optimum. Two corrections to the original sketch came out of that check: Beta Resentful (Grim Trigger) never needs a switch, plain Alpha is optimal; and Tat For Tit, Detective and Pavlov now have counters (Beta every round, Beta then Alpha, Alpha every round).

---

## Tuning the prompts

Everything the model reads is in [`prompts.toml`](prompts.toml); `config.toml` only holds experiment settings.

* `[blocks.*]`: reusable paragraphs (`rules`, `goal`, `mechanics`, `tips`, horizon variants, ...). Edit one and every tier that uses it changes.
* `[tiers.*]`: an ordered list of block names. Add a new tier by adding a section; reorder or drop blocks to ablate them.
* `[turn]`, `[reflection]`, `[dreamer]`: the per-round, post-match and consolidation templates.

A typo in a `{placeholder}` or block name fails at start-up with a clear message, before any tokens are spent.

**A tuning loop that tells you what to change:**

1. `python main.py --show-prompt <tier>` to read exactly what the model sees.
2. Run, then open `scorecard_training.md`. Find the opponent with the lowest **efficiency** (points earned divided by the best achievable against that opponent) and read the **verdict** lines.
3. Open `trace.jsonl` and look at the turns against that opponent: the stored prompt, the model's hypothesis and reasoning. Typical diagnoses:
   * Alpha rate near 0% and low efficiency across the board: the model is stuck in Beta/Beta. Strengthen the `mechanics` block.
   * High efficiency against Always Alpha-type opponents but low against reciprocators: it does not retaliate-proof its play; look at `tips`.
   * Hypotheses never name the right strategy (identification near 0%): in `t3` the catalog descriptions are unclear; in `t4`/`t5` expect this, compare with `t3`.
   * Lessons in `memory.json` contradict the scorecard: the reflection prompt is letting wrong rules in; tighten point 3 of `[reflection]`.
4. Edit one block, rerun with the same `seed`, compare scorecards. `--tiers a,b` does a side-by-side run.

---

## Is it working? The scorecard

Every training run and tournament writes `scorecard_*.md/.json` and prints it at the end. The training run also shows a **live status line** under the progress bars (running points per round, efficiency, Alpha rate, fallbacks, parse errors).

Per opponent it reports:

* **Points per round** and **efficiency**: points earned divided by the best score any fixed move sequence can get against that opponent (found by exhaustive search for matches up to 10 rounds; Random uses its analytic optimum of 3.0 per round).
* **Alpha rate** and **mutual Alpha rate**.
* **Identification accuracy** and rounds to identify: how often the model's stated hypothesis names the true opponent (not shown for `t1`/`t2`, where the opponent is given).
* **LLM health**: turns, retries, fallbacks, parse failures, failed reflections, rate-limit hits.
* **Verdicts**: `PASS`/`OK`/`WARN`/`FAIL` lines, for example efficiency below 70% against an opponent, or "stuck in Beta/Beta", or "N rounds were played by the fallback rule".

Results from API trouble and results from bad play are kept apart on purpose: a failed call never silently becomes a Beta.

## Why did a run fail? Diagnosing

| Question | Where to look |
|---|---|
| Did the API fail? | `scorecard_*.md` "LLM health" and `FAIL`/`WARN` lines; `trace.jsonl` lines with `"failed": true` or `kind: fallback` hold the exception text |
| What exactly did the model see and say? | `trace.jsonl`: one line per LLM call with the full prompt, raw response, parsed result, attempts, errors, latency, tokens |
| Did the model play the opponent well? | per-opponent efficiency in the scorecard; `training_history.json` has every round's moves (`"CD"` = you Alpha, they Beta) and hypothesis |
| What did it learn? | `memory.json` (observations and lessons with importance and evidence links), `master_strategy_playbook.md` |
| Did the playbook fall back to a data summary? | `master_strategy_playbook.json` has `"fallback": true` |
| Why did it stop? | the console prints `RUN STOPPED: ...` (exit code 2) and the checkpoint stays on disk |

Behaviour on failure: each call is retried (`llm.max_retries`); if all attempts fail the round uses a **neutral fallback** (Alpha on round 1, then repeat the opponent's last move; it is counted and traced); after `llm.max_consecutive_fallbacks` failures in a row the run stops instead of producing garbage; three consecutive 429s stop it as a quota abort.

What went wrong in the old failed run (`output_001`), found while reading its outputs: the model collapsed into Beta/Beta (about 1 point per round against Suspicious Tit For Tat and under 2 against Tit For Two Tats, where mutual Alpha pays 3 or more), and its reflections then repeated the same wrong rule about 40 times ("defect, enforce the Nash equilibrium"), which the Dreamer turned into a playbook. The prompt called the game by name, used "Nash" and "score differential" framing, and only told the model what *not* to say. Failures were also printed once and not stored, so the cause could not be recovered. The new prompts teach the point arithmetic of repeated play and use neutral wording, memory de-duplicates and ranks lessons, the Dreamer is given measured points per round (and told to drop lessons the numbers contradict), and every failure is traced.

---

## Architecture

```
prompts.toml  ──┐        config.toml
(blocks, tiers) │             │
                ▼             ▼
            main.py ── run_pipeline(tier)
                │
     ┌──────────┼───────────────────────────────┐
     ▼          ▼                               ▼
 TrainingBattery ──► Dreamer ──► playbook ──► RoundRobinTournament
     │  plays via         │ uses                    │ plays via
     ▼                    ▼                         ▼
 match_runner.play_match (axelrod.Match)  ◄──  LLMPlayer (an axelrod.Player)
                                                │ memory stream, trace, retries, usage tracker
                                                ▼
                               scorecard.py ──► scorecard_*.md/json, live TUI status
```

| Module | Role |
|---|---|
| `src/opponents.py` | Opponent registry on the **Axelrod** library (`axl.Cooperator`, `axl.TitForTat`, `axl.TitFor2Tats`, `axl.AntiTitForTat`, `axl.Detective`, `axl.Grudger`, `axl.WinStayLoseShift`, ...), neutral names, descriptions, verified counters |
| `src/match_runner.py` | `play_match` on `axelrod.Match` (payoffs 3/0/5/1), seeded match-length sampling (fixed, Normal, or geometric/unknown) |
| `src/llm_player.py` | The LLM as an `axl.Player`: retries, fallback, tracing, token accounting, `mock` provider, reflection |
| `src/memory.py` | Memory stream |
| `src/prompts.py` | Loads `prompts.toml`, validates it, renders prompts |
| `src/dreamer.py` | Consolidation into a playbook from measured results and lessons |
| `src/training_battery.py`, `src/tournament.py` | The two phases, both checkpointed and resumable |
| `src/scorecard.py` | Efficiency, identification, health, verdicts, tier comparison |
| `src/trace.py`, `src/usage_tracker.py` | JSONL trace of every call; request and token counters (written to the run's output dir) |

### Memory (after Park et al. 2023, *Generative Agents*)

Using the paper's core ideas without embeddings, so it works with small models:

* **Memory stream** of entries. *Observations* are facts written by the harness after each match (moves, points per round). *Reflections* are lessons the model writes, each with an LLM-rated **importance** (1-10) and **evidence** links to the observation they came from.
* **Retrieval** ranks entries by recency (decays per match) + importance + relevance (keyword overlap with the model's latest hypothesis; the opponent tag is used only in tiers that reveal the opponent). Only the top `memory_top_k` go into each prompt, so memory cannot grow the prompt without limit.
* **De-duplication**: restating a lesson raises the importance of the existing one instead of adding a copy.
* **Dreamer = planning step**: it is given measured results per opponent (points per round, Alpha rate, best sequence) next to the lessons, and must drop lessons the numbers contradict. The playbook is added to the retrieved memories rather than replacing them.

---

## Configuration (`config.toml`)

Unknown keys are errors. The main settings:

```toml
[experiment]  seed = 42, output_dir = "output"          # seed fixes match lengths and the Random opponent
[llm]         temperature, max_retries, max_consecutive_fallbacks
[llm.rate_limit]  max_rpm, max_rpd
[environment] error_rate = 0.0                           # chance a move is flipped
[prompts]     tier = "t3_pool_hint", prompts_file, memory_top_k
[training]    matches_per_strategy, rounds_per_match, strategies = [...]
[training.dreamer]  enabled, temperature
[evaluation]  matches_per_pairing, rounds_per_match, participants = [...], [evaluation.scoring]
[reporting]   export_html / export_markdown / export_json
```

`rounds_per_match` accepts an integer, `{ average, std_dev }` (bell curve), or `{ prob_end = 0.1 }` (after every round the match ends with that probability: the unknown-horizon setting for `t5`).

Available opponent keys: `AlwaysCooperate, AlwaysDefect, TitForTat, SuspiciousTitForTat, TitForTwoTats, TatForTit, Random, Detective, GrimTrigger, Pavlov`. Adding another Axelrod strategy is one `OpponentSpec` entry in `src/opponents.py`.

CLI flags: `--tier`, `--tiers a,b|all`, `--provider`, `--model`, `--seed`, `--quick`, `--train-only`, `--eval-only`, `--no-llm`, `--show-prompt TIER [--opponent KEY]`, `--output-dir`, `--config`.

---

## Outputs

In `output_dir` (or `output_dir/<tier>/` with `--tiers`):

| File | Contents |
|---|---|
| `scorecard_training.md/.json`, `scorecard_tournament.md/.json` | the scorecard |
| `trace.jsonl` | every LLM call: prompt, raw response, parse result, errors, tokens, latency |
| `training_history.json` | per match: moves, per-round hypotheses, scores, new lessons, LLM health deltas |
| `memory.json`, `master_strategy_playbook.md/.json` | what was learned (`"fallback"` flag in the JSON) |
| `tournament_leaderboard.md/.json`, `tournament_report.html` | standings, cross-table, per-match moves |
| `api_usage.json` | request timestamps and token totals |
| `tier_comparison.md/.json` | only with `--tiers`: one row per tier |
| `training_checkpoint.json`, `tournament_checkpoint.json` | present only while a run is unfinished |

---

## Testing and what has been validated

```powershell
pytest          # 102 tests, about 20 seconds, no network
```

* **Opponents and counters**: each opponent against scripted move sequences; each `t1` counter against the brute-force optimum; neutral wording enforced.
* **Prompts**: all tiers render with no stray placeholders; the information ladder (what each tier does and does not reveal); no leaks of the opponent in blind tiers; typos in `prompts.toml` fail fast.
* **LLM player**: against a stubbed client: retries, neutral fallback, abort after repeated failures, 429 abort, trace contents, token accounting, memory surviving between matches, frozen behaviour.
* **Memory and scorecard**: de-duplication, ranking, regret/efficiency numbers, the Beta/Beta collapse being flagged, fallbacks reported apart from play quality.
* **Pipeline** (offline `mock` provider): full run writes every artifact; training resumes after a mid-run LLM outage without repeating matches; corrupt checkpoint is reported; tournaments are reproducible from a seed and resume to identical results; `--tiers all`, `--eval-only`, `--show-prompt`, clean stop with exit code 2.
* **A real model**: the complete pipeline ran on a local `qwen2.5:3b` (Ollama) for all five tiers (`--tiers all`, 3 strategies x 3 matches x 6 rounds, seed 11): 54 training turns per tier, **0 retries, 0 fallbacks**, so every number below reflects the model, not the harness.

  | Tier | Tournament efficiency | Alpha rate | Identification |
  |---|---|---|---|
  | `t1_informed` | 55% | 85% | n/a |
  | `t2_named` | 49% | 100% | n/a |
  | `t3_pool_hint` | 49% | 100% | 33% |
  | `t4_blind` | 49% | 100% | 0% |
  | `t5_open_ended` | 49% | 100% | 0% |

  What this shows: the scorecard tells the tiers apart. Only `t1` (which is handed the counters) changed behaviour at all, and only partly: it exploited Always Alpha at 78-87% efficiency but ignored "play Beta every round" against Tat For Tit and stayed at 10%. Only `t3` produced any correct identification (39-56% in training). In `t2`-`t5` this 3B model plays Alpha every round regardless, the opposite failure from the old run's Beta collapse, and the scorecard flags it (60% efficiency against Always Alpha, 10% against Tat For Tit). With 3 matches per opponent and one seed, differences of a few points are not significant; the pattern (only `t1` moves, only `t3` identifies) is the finding.

**Not yet validated**: whether the new prompts beat the old ones on a capable model, and the comparison between tiers at a meaningful sample size. Both need runs with more matches (`matches_per_strategy = 5` or more) on a stronger model; `--tiers all` and the scorecard are built for exactly that. LLM output is not deterministic even at low temperature, so repeat runs before drawing conclusions.

---

## Notes

* `docs/LLM Instructions.md` is the original sketch the tiers were built from; `prompts.toml` is the maintained version.
* The Axelrod library is also available for a much larger opponent pool (`axl.all_strategies`) if you want to test generalisation beyond the ten hand-picked strategies.
* `output/`, `output_000/` and `output_001/` are results of earlier versions and are git-ignored.
