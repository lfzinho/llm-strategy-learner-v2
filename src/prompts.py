from typing import Dict, List, Optional, Tuple

MATCH_LENGTH_DESCRIPTION = "Each match is Normal(10, 1) rounds long; the exact number of rounds is not revealed in advance."

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


def format_known_strategies(known_strategies: Optional[List[str]] = None) -> str:
    known = known_strategies or list(STRATEGY_DESCRIPTIONS.keys())
    desc_lines = []
    for name in known:
        desc = STRATEGY_DESCRIPTIONS.get(name, "Algorithmic strategy.")
        desc_lines.append(f"  • {name}: {desc}")
    return "\n".join(desc_lines)


def build_system_prompt(
    mode: str = "explicit_strategies",
    custom_system_prompts: Optional[Dict[str, str]] = None,
    known_strategies: Optional[List[str]] = None,
    opponent_strategy: Optional[str] = None,
) -> str:
    mode_key = mode.lower()
    known_desc = format_known_strategies(known_strategies)

    # Check if a custom system prompt was defined in config.toml
    if custom_system_prompts and mode_key in custom_system_prompts and custom_system_prompts[mode_key].strip():
        template = custom_system_prompts[mode_key]
        return (
            template
            .replace("{known_strategies_descriptions}", known_desc)
            .replace("{opponent_strategy}", opponent_strategy or "Unknown")
        )

    # Built-in fallback templates
    if mode_key == "explicit_strategies":
        return f"""You are an expert game theory agent playing the Iterated Prisoner's Dilemma.

ENVIRONMENT & RULES:
- The environment is completely deterministic (0% error rate). Every move you and your opponent make executes exactly as intended.
- {MATCH_LENGTH_DESCRIPTION}
- Payoff Matrix:
  * Both Cooperate: 3 points each
  * You Defect, Opponent Cooperates: You get 5 points, Opponent gets 0
  * You Cooperate, Opponent Defects: You get 0 points, Opponent gets 5
  * Both Defect: 1 point each

STRATEGY IDENTIFICATION TASK (KNOWN POOL):
You are competing against an opponent using one of the following classical strategies:
{known_desc}

YOUR PRIMARY OBJECTIVE:
1. Infer the opponent's decision rule from the observed move history. The strategy descriptions are clues for diagnosis, not answers for your counter-policy.
2. Choose the move that maximizes YOUR TOTAL CUMULATIVE SCORE over the remaining rounds. Score differential is secondary and must never replace your own score as the objective.
3. Independently derive the policy from the payoff matrix and the opponent's observed transition rule. Compare candidate continuations: mutual cooperation pays 3 each round, mutual defection pays 1 each round, exploiting cooperation pays 5 for one round but may trigger retaliation, and cooperating against defection pays 0.
4. Balance immediate payoff against future consequences in this repeated, finite-horizon game. Do not call mutual defection optimal merely because it is a one-shot Nash equilibrium.
5. Explain the policy you derive from behavioral evidence and payoffs. Do not use a memorized strategy-to-response table.
"""

    elif mode_key == "generic_archetypes":
        return """You are an expert game theory agent playing the Iterated Prisoner's Dilemma.

ENVIRONMENT & RULES:
- The environment is completely deterministic (0% error rate). Every move you and your opponent make executes exactly as intended.
- Each match is Normal(10, 1) rounds long; the exact number of rounds is not revealed in advance.
- Payoff Matrix:
  * Both Cooperate: 3 points each
  * You Defect, Opponent Cooperates: You get 5 points, Opponent gets 0
  * You Cooperate, Opponent Defects: You get 0 points, Opponent gets 5
  * Both Defect: 1 point each

STRATEGY IDENTIFICATION TASK (CLASSICAL HEURISTICS):
You are competing against opponents who follow fixed classical game-theoretic strategies. 
Opponents may be purely cooperative, purely exploitative, strictly reciprocal, forgiving, probing/testing, or randomized.

YOUR PRIMARY OBJECTIVE:
1. Rapidly formulate a hypothesis about what decision rule or heuristic your opponent follows.
2. Test their reaction to your moves in the opening rounds without assuming that defection is inherently safer.
3. Optimize YOUR TOTAL CUMULATIVE SCORE dynamically. Compare the immediate payoff and expected future payoff of cooperation and defection using the observed transition rule.
4. Treat mutual defection as one possible outcome, not as a universal objective or proof of optimality.
5. Explain the policy you independently derive from the observed behavior and payoff matrix.
"""

    elif mode_key == "blind_optimization":
        return """You are an expert game theory agent playing the Iterated Prisoner's Dilemma.

ENVIRONMENT & RULES:
- The environment is completely deterministic (0% error rate). Every move you and your opponent make executes exactly as intended.
- Each match is Normal(10, 1) rounds long; the exact number of rounds is not revealed in advance.
- Payoff Matrix:
  * Both Cooperate: 3 points each
  * You Defect, Opponent Cooperates: You get 5 points, Opponent gets 0
  * You Cooperate, Opponent Defects: You get 0 points, Opponent gets 5
  * Both Defect: 1 point each

STRATEGY OPTIMIZATION TASK:
Your goal is to maximize YOUR TOTAL CUMULATIVE SCORE in the Iterated Prisoner's Dilemma. Do not optimize score differential as a substitute objective.
Observe the opponent's actions across rounds, infer the rule mapping history to future moves, and independently derive the best response from the payoff matrix. Compare cooperation and defection over the remaining horizon; do not assume that the one-shot Nash equilibrium remains the best iterated policy.
"""
    elif mode_key == "revealed_strategy":
        return """You are an expert game theory agent playing the Iterated Prisoner's Dilemma.

ENVIRONMENT & RULES:
- The environment is completely deterministic (0% error rate). Every move you and your opponent make executes exactly as intended.
- Each match is Normal(10, 1) rounds long; the exact number of rounds is not revealed in advance.
- Your opponent is playing this strategy: {opponent_strategy}

STRATEGY RESPONSE TASK:
Use the revealed strategy identity only as a hypothesis about behavior. Verify it against observed moves, then independently derive the mathematically optimal counter-strategy from the payoff matrix and remaining horizon. Maximize YOUR TOTAL CUMULATIVE SCORE; score differential is secondary.
""".format(opponent_strategy=opponent_strategy or "Unknown")
    else:
        raise ValueError(f"Unknown prompt mode: '{mode}'. Available: explicit_strategies, generic_archetypes, blind_optimization, revealed_strategy, custom")


def build_turn_prompt(
    game_history: List[Tuple[Move, Move]],
    my_score: int,
    opp_score: int,
    long_lasting_notes: List[str],
    short_lasting_notes: List[str],
    master_playbook: Optional[str] = None,
    mode: str = "explicit_strategies",
    custom_system_prompts: Optional[Dict[str, str]] = None,
    known_strategies: Optional[List[str]] = None,
    opponent_strategy: Optional[str] = None,
) -> str:
    system_intro = build_system_prompt(mode, custom_system_prompts, known_strategies, opponent_strategy)
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
- Match length: Normal(10, 1) rounds (exact length not revealed in advance)
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
    custom_template: Optional[str] = None,
) -> str:
    history_text = format_history_lines(game_history)
    short_notes_text = format_notes(short_lasting_notes)
    long_notes_text = format_notes(long_lasting_notes)
    revealed_opp = f"(Revealed Opponent Identity: {actual_opponent_name})" if actual_opponent_name else ""

    if custom_template and custom_template.strip():
        return (
            custom_template
            .replace("{revealed_opponent_info}", revealed_opp)
            .replace("{my_score}", str(my_score))
            .replace("{opp_score}", str(opp_score))
            .replace("{history_text}", history_text)
            .replace("{short_notes_text}", short_notes_text)
            .replace("{long_notes_text}", long_notes_text)
        )

    return f"""You just finished a Normal(10, 1)-round match of Iterated Prisoner's Dilemma. {revealed_opp}
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


def build_dreamer_prompt(
    training_summary: str,
    raw_notes: List[str],
    custom_template: Optional[str] = None,
) -> str:
    raw_notes_text = format_notes(raw_notes)

    if custom_template and custom_template.strip():
        return (
            custom_template
            .replace("{training_summary}", training_summary)
            .replace("{raw_notes_text}", raw_notes_text)
        )

    return f"""You are the Strategy Synthesis Dreamer for an autonomous game-theoretic LLM agent.
The agent has just completed a comprehensive structured training battery against various classical opponents in the Iterated Prisoner's Dilemma.

TRAINING SUMMARY & RESULTS:
{training_summary}

RAW NOTES & OBSERVATIONS ACCUMULATED DURING TRAINING:
{raw_notes_text}

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
