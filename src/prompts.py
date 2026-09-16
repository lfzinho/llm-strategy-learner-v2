from typing import List, Optional, Tuple

try:
    from src.constants import Move
    from src.player_classes import STRATEGY_DESCRIPTIONS
except ImportError:
    from constants import Move
    from player_classes import STRATEGY_DESCRIPTIONS


def format_history_lines(game_history: List[Tuple[Move, Move]]) -> str:
    if not game_history:
        return "This is the first round against this opponent. No prior moves."
    lines = []
    for i, (my_m, opp_m) in enumerate(game_history, 1):
        my_str = "COOPERATE" if my_m == Move.COOPERATE else "DEFECT"
        opp_str = "COOPERATE" if opp_m == Move.COOPERATE else "DEFECT"
        lines.append(f"Round {i}: You played {my_str}, Opponent played {opp_str}")
    return "\n".join(lines)


def format_notes(notes: List[str]) -> str:
    if not notes:
        return "- (None)"
    return "\n".join(f"- {note}" for note in notes)


def get_mode_instructions(mode: str, known_strategies: Optional[List[str]] = None) -> str:
    mode = mode.lower()
    if mode == "explicit_strategies":
        known = known_strategies or list(STRATEGY_DESCRIPTIONS.keys())
        desc_lines = []
        for name in known:
            desc = STRATEGY_DESCRIPTIONS.get(name, "Algorithmic strategy.")
            desc_lines.append(f"  • {name}: {desc}")
        desc_text = "\n".join(desc_lines)

        return f"""STRATEGY IDENTIFICATION TASK (KNOWN POOL):
You are competing against an opponent using one of the following classical strategies:
{desc_text}

YOUR PRIMARY OBJECTIVE:
1. Identify the opponent's strategy as early as possible in the match (within rounds 1-2). Use opening moves to test/probe their reaction if necessary.
2. Once the opponent's strategy is identified, execute the mathematically optimal counter-play to maximize your cumulative score:
   - Against unconditional cooperators (Always Cooperate): Defect to score maximum points (5 per round).
   - Against unconditional defectors (Always Defect): Defect to avoid exploitation (1 per round vs 0).
   - Against reciprocal players (Tit For Tat, Tit For Two Tats): Cooperate to secure mutual cooperation (3 per round).
   - Against probing players (Detective): Respond firmly to probes so they settle into mutual cooperation.
"""

    elif mode == "generic_archetypes":
        return """STRATEGY IDENTIFICATION TASK (CLASSICAL HEURISTICS):
You are competing against opponents who follow fixed classical game-theoretic strategies. 
Opponents may be purely cooperative, purely exploitative, strictly reciprocal, forgiving, probing/testing, or randomized.

YOUR PRIMARY OBJECTIVE:
1. Rapidly formulate a hypothesis about what decision rule or heuristic your opponent follows.
2. Test their reaction to your moves in the opening rounds.
3. Optimize your play dynamically once you detect whether they are forgiving, retaliatory, exploitative, or random.
"""

    elif mode == "blind_optimization":
        return """STRATEGY OPTIMIZATION TASK:
Your goal is to maximize your cumulative score in the Iterated Prisoner's Dilemma.
Observe the opponent's actions across rounds, detect behavioral patterns, and choose the moves that yield the highest possible long-term payoff.
"""
    else:
        raise ValueError(f"Unknown prompt mode: {mode}")


def build_system_prompt(
    mode: str = "explicit_strategies",
    known_strategies: Optional[List[str]] = None,
) -> str:
    mode_text = get_mode_instructions(mode, known_strategies)
    return f"""You are an expert game theory agent playing the Iterated Prisoner's Dilemma.

ENVIRONMENT & RULES:
- The environment is completely deterministic (0% error rate). Every move you and your opponent make executes exactly as intended.
- Payoff Matrix:
  * Both Cooperate: 3 points each
  * You Defect, Opponent Cooperates: You get 5 points, Opponent gets 0
  * You Cooperate, Opponent Defects: You get 0 points, Opponent gets 5
  * Both Defect: 1 point each

{mode_text}
"""


def build_turn_prompt(
    game_history: List[Tuple[Move, Move]],
    my_score: int,
    opp_score: int,
    long_lasting_notes: List[str],
    short_lasting_notes: List[str],
    master_playbook: Optional[str] = None,
    mode: str = "explicit_strategies",
    known_strategies: Optional[List[str]] = None,
) -> str:
    system_intro = build_system_prompt(mode, known_strategies)
    history_text = format_history_lines(game_history)
    short_notes_text = format_notes(short_lasting_notes)

    playbook_section = ""
    if master_playbook and master_playbook.strip():
        playbook_section = f"""
MASTER STRATEGY PLAYBOOK (Synthesized meta-memory from training):
{master_playbook.strip()}
"""
    else:
        long_notes_text = format_notes(long_lasting_notes)
        playbook_section = f"""
LONG-TERM STRATEGY NOTES:
{long_notes_text}
"""

    round_number = len(game_history) + 1

    return f"""{system_intro}
MATCH STATUS:
- Round: {round_number}
- Your Current Match Score: {my_score}
- Opponent Current Match Score: {opp_score}

MOVE HISTORY SO FAR:
{history_text}
{playbook_section}
SHORT-TERM MEMORY (Current match observations):
{short_notes_text}

TASK:
1. Formulate your hypothesis on the opponent's strategy based on their move pattern.
2. Determine your optimal next move: COOPERATE or DEFECT.
3. Provide concise reasoning explaining why this move maximizes your score given your hypothesis.
4. Optionally provide a new short-term memory note if you made a key observation about this opponent.
"""


def build_reflection_prompt(
    game_history: List[Tuple[Move, Move]],
    my_score: int,
    opp_score: int,
    short_lasting_notes: List[str],
    long_lasting_notes: List[str],
    actual_opponent_name: Optional[str] = None,
) -> str:
    history_text = format_history_lines(game_history)
    short_notes_text = format_notes(short_lasting_notes)
    long_notes_text = format_notes(long_lasting_notes)

    revealed_opp = f"(Revealed Opponent Identity: {actual_opponent_name})" if actual_opponent_name else ""

    return f"""You just finished a 5-round match of Iterated Prisoner's Dilemma. {revealed_opp}
Final Score: You = {my_score} pts | Opponent = {opp_score} pts

MATCH MOVE HISTORY:
{history_text}

OBSERVATIONS RECORDED DURING MATCH:
{short_notes_text}

EXISTING PERSISTENT STRATEGY NOTES:
{long_notes_text}

POST-MATCH REFLECTION TASK:
1. What strategy/archetype did this opponent use? How did they react to your opening and subsequent moves?
2. Did your play achieve the maximum possible payoff against this specific strategy? If not, what would have been mathematically superior?
3. Formulate a new, concise long-term heuristic or archetype identification rule to remember for all future games against this type of opponent. (If no new rule is needed, you may leave it empty).
"""


def build_dreamer_prompt(training_summary: str, raw_notes: List[str]) -> str:
    notes_formatted = format_notes(raw_notes)
    return f"""You are the Strategy Synthesis Dreamer for an autonomous game-theoretic LLM agent.
The agent has just completed a comprehensive structured training battery against various classical opponents in the Iterated Prisoner's Dilemma.

TRAINING SUMMARY & RESULTS:
{training_summary}

RAW NOTES & OBSERVATIONS ACCUMULATED DURING TRAINING:
{notes_formatted}

YOUR TASK:
Consolidate, distill, and synthesize all the raw notes and match experiences into a single, cohesive, high-density MASTER STRATEGY PLAYBOOK.
This playbook will be frozen into the agent's memory for the upcoming championship tournament.

The Master Strategy Playbook MUST include:
1. Quick Archetype Recognition Guide:
   - What to play in Round 1 and Round 2 to probe and reliably diagnose the opponent's strategy.
   - Decision tree / signature behavioral markers for each archetype (Always Cooperate, Always Defect, Tit For Tat, Suspicious TFT, Tit For Two Tats, Detective, Tat For Tit, Random, etc.).
2. Optimal Counter-Strategy Policy:
   - For each recognized archetype, the exact optimal sequence of moves to maximize score.
   - When to cooperate for mutual 3-3 payoff, and when to defect for 5-0 payoff.
3. Universal Principles & Defensive Safeguards:
   - How to avoid costly feuds or sucker payoffs.
   - End-game moves.

Output the synthesized playbook in clear, dense, actionable Markdown.
"""
