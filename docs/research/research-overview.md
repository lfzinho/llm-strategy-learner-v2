# How do LLM agents behave in repeated social dilemmas with communication and reputation?

*Research overview and Step 1 report. Status: Step 1 complete, Steps 2 and 3 to be defined.*

---

## 1. The problem

A social dilemma is a situation in which each participant does better by acting selfishly, yet everyone does worse if all do. The Prisoner's Dilemma (PD) is the simplest case. Two players each choose to cooperate or defect. Defecting pays more whatever the other does, but mutual cooperation beats mutual defection. When the game is played once, defection is the rational choice. When it is **repeated**, cooperation can be sustained, because today's betrayal can be punished tomorrow. Axelrod's tournaments (1980s) showed that simple reciprocal strategies such as Tit For Tat did very well in this repeated setting.

Large language models (LLMs) are now being deployed as agents that negotiate, trade and collaborate, with people and with each other. Whether they cooperate, retaliate, forgive, deceive or free-ride is therefore a practical question as well as a theoretical one. It is also hard to answer with a quick test, because an LLM's behaviour depends on the wording of the prompt, the information it is given, what it remembers, and whether it can talk to the other side.

**Central question.** *How do LLM agents behave in repeated social dilemmas, and how does that behaviour change when we add (a) communication and (b) reputation?*

We study this in three steps, each adding one source of social complexity to the one before.

| Step | Setting | What is new | Status |
|---|---|---|---|
| 1 | One LLM agent against fixed, classical strategies | A measurable baseline; how information and memory affect play | **Done (this document)** |
| 2 | Several LLM agents playing each other, with messages before each round | Communication ("cheap talk"), possible deception | TBD |
| 3 | A population of agents with a public record of who betrayed whom | Reputation and third-party observation | TBD |

The ordering is deliberate. Step 1 gives a controlled setting where the optimal play is known, so any weakness of the LLM can be measured exactly. Steps 2 and 3 remove that ground truth in exchange for realism, so we want to arrive there with a trusted measurement toolkit and a known baseline of how a single agent behaves.

---

## 2. Step 1: a single LLM agent against classical strategies

### 2.1 Goal

Build a system that lets an LLM learn a strategy for the iterated PD, and measure precisely how well it plays and why it fails when it does. The specific questions are:

1. How close to the best achievable score does an LLM get against known opponents?
2. How much does it matter what the model is told about its opponent and about the length of the game?
3. Can the model identify the opponent from the play alone?
4. Does a learning pipeline (training matches, reflection, a distilled playbook) help?

### 2.2 The game

Players choose between two actions, shown to the model with neutral labels, **Alpha** (cooperate) and **Beta** (defect), so that the words "Prisoner's Dilemma", "cooperate" and "defect" do not trigger memorised behaviour. Payoffs per round (own points / opponent points):

| | Opponent Alpha | Opponent Beta |
|---|---|---|
| **I play Alpha** | 3 / 3 | 0 / 5 |
| **I play Beta** | 5 / 0 | 1 / 1 |

Matches last about 10 rounds, drawn from a normal distribution with mean 10 and standard deviation 1. In the unknown-horizon tier, the length is geometric (each round ends the game with probability 0.1), so the model cannot plan around a fixed last round.

### 2.3 Opponents

Matches and opponents run on the **Axelrod library**, the standard open-source framework for the iterated PD, instead of hand-written game code. The pool has 10 strategies, covering the main behavioural types:

| Our name | Axelrod class | Behaviour |
|---|---|---|
| Always Alpha | `Cooperator` | Always cooperates |
| Always Beta | `Defector` | Always defects |
| Alpha Mimic | `TitForTat` | Starts cooperating, copies the last move |
| Beta Mimic | `SuspiciousTitForTat` | Starts defecting, then copies |
| Zen Alpha Mimic | `TitFor2Tats` | Retaliates only after two defections |
| Reverse Mimic | `AntiTitForTat` | Plays the opposite of the last move |
| Random Choice | `Random` | 50/50 |
| Prober | `Detective` | Tests with a fixed opening, then exploits or mimics |
| Beta Resentful | `Grudger` | Cooperates until betrayed once, then defects forever |
| Stay-Shift | `WinStayLoseShift` | Repeats a move that paid well, switches otherwise |

For each opponent we computed the **best achievable score** by exhaustive search over fixed move sequences (up to 10 rounds), with the analytic value for Random (3.0 per round). The exact counter-strategies were checked against brute force in the test suite.

### 2.4 The five prompt tiers

All prompts are assembled from reusable blocks in a single file, `prompts.toml`, so a tier is a list of blocks and nothing else. The tiers form an information ladder:

| Tier | Opponent | Counter-strategy | Horizon |
|---|---|---|---|
| `t1_informed` | named, with description | given | exact length and end-game hint |
| `t2_named` | named | not given | exact length |
| `t3_pool_hint` | hidden; the list of possible opponents is shown | not given | exact length |
| `t4_blind` | hidden, no list | not given | exact length |
| `t5_open_ended` | hidden, no list | not given | unknown (geometric) |

The ladder is the main experimental variable. Moving from `t1` down to `t5` removes one kind of information at a time.

### 2.5 The learning pipeline

The pipeline has three phases.

1. **Training battery.** The agent plays several matches against each opponent. After each match it writes an *observation* (facts measured by the harness: what the opponent actually did) and a *reflection* (a lesson, rated 1 to 10 for importance, citing its evidence).
2. **Dreamer.** A separate call reads the measured results of the whole training phase and writes a *playbook*: for each opponent type, how to recognise it and what to do. This plays the role of the planning step in Park et al.'s generative-agents architecture, but is grounded in measured scores instead of the model's unsupported impressions.
3. **Tournament.** The agent plays a round-robin against every opponent with the playbook and its memory frozen.

The memory follows the core of Park et al. (2023) without embeddings: entries are retrieved by a score combining recency, importance and keyword relevance, capped at *k* entries, and near-duplicates are merged. This addresses a failure we observed in the original project, where the same wrong lesson was written about 40 times and then reinforced itself.

### 2.6 Measurement and observability

A weakness of the original code was that failures were invisible: any error silently became a default move. The new system records:

- a **trace** (`trace.jsonl`) of every LLM call: prompt, raw response, parsed result, retries, errors;
- an explicit **fallback** policy, with a counter and an abort after a configurable number of consecutive failures, so a run cannot quietly degrade into noise;
- **checkpoints** for training, the Dreamer and the tournament, so a failed run resumes without repeating matches;
- a **scorecard** per run: points per round, *efficiency* (points earned divided by the best achievable), Alpha rate, mutual-Alpha rate, identification accuracy (does the model's per-round guess match the true opponent?), call health, and PASS/OK/WARN/FAIL verdicts.

A suite of 102 offline tests (using a deterministic `mock` provider and a stubbed client) covers the opponents, prompts, memory, scorecard, retry/fallback behaviour, checkpoint recovery and the full pipeline.

### 2.7 Results

**Capable model.** `anthropic/claude-haiku-5.5` through OpenRouter, all five tiers, the 8-opponent pool, 3 matches per strategy and per pairing, seed 42, 0 fallbacks, total cost about $3.34.

| Tier | Training efficiency | Tournament efficiency | Points/round | Identification |
|---|---|---|---|---|
| `t1_informed` | 99% | 99% | 3.42 | n/a |
| `t2_named` | 98% | 100% | 3.43 | n/a |
| `t3_pool_hint` | 86% | 90% | 3.10 | 41% |
| `t4_blind` | 85% | 84% | 2.91 | 7% |
| `t5_open_ended` | 79% | 76% | 2.64 | 8% |

**Local small models** (Ollama, 3 tiers or more, same seed): `qwen2.5:3b` scored 49-55% and `dolphin-llama3` (8B) scored 49-88%. They play Alpha almost every round regardless of the opponent and change behaviour only when the counter-strategy is written in the prompt.

**Findings.**

1. *Performance falls steadily as information is removed.* Told who the opponent is, the capable model plays essentially optimally, including exploiting naive opponents (5.0 points per round against Always Alpha). Without the name it loses 10 to 25 points of efficiency.
2. *Identification from play alone is poor*, even for the capable model (7-8% without a list of candidates, 41% with one). Blind-tier results therefore reflect generally sensible behaviour, not recognition of the opponent.
3. *The hardest opponent is the forgiving one* (Zen Alpha Mimic): without its name, efficiency drops to 61% in `t4`. The scorecard shows the loss but we have not yet analysed the traces to say why (a hypothesis to test: it does not discover the alternating exploit).
4. *An unknown horizon makes the model more cautious*: it earns only 60% of the possible score against Always Alpha in `t5`.
5. *The original failure did not recur.* The earlier version collapsed into mutual defection, reinforced by repeated "equilibrium" lessons. No tier of the new system showed it, and the scorecard flags it automatically if it returns.
6. *Small models fail in the opposite direction*: they over-cooperate, and the scorecard identifies this as well.

### 2.8 Limitations

- Only 3 matches per opponent, one seed and one model for the capable run. Gaps of a few points between neighbouring tiers are noise; only the overall pattern is reliable.
- The old prompts were never run on the same model, so there is no head-to-head showing the new system is better, only the absence of the old failure.
- The opponent pool is small and deterministic (apart from Random). The playbook may be partly memorising it. A held-out set of unseen strategies is needed to test generalisation.
- The best-achievable value is exact only up to 10 rounds; beyond that it is a lower bound.
- LLM output is not deterministic even at low temperature.

### 2.9 What Step 1 provides for the later steps

- A tested harness with tracing, checkpointing and cost tracking.
- A scorecard methodology and the idea of *efficiency against a known optimum*, which Steps 2 and 3 cannot use directly but can be compared against.
- The prompt-block system, so that adding a "messages" or "reputation" block is a configuration change.
- A baseline: single-agent behaviour against fixed opponents, by model and by information tier.
- The memory stream, which Steps 2 and 3 will need so that agents can remember who said and did what.

---

## 3. Step 2: LLM agents playing each other, with communication

**TBD.**

## 4. Step 3: a population with reputation

**TBD.**

---

## Where the files are

- `README.md` (repository root): how to run the system and the full validation record.
- `docs/research/literature-review.md`: related work and how each source can be used.
- `prompts.toml`: all prompt text and tier definitions.
- `config.toml`: experiment settings.
