import argparse
import json
import os
import sys

from dotenv import load_dotenv

# Ensure Windows stdout handles UTF-8 (unicode tables)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

load_dotenv()

from src.config import MasterConfig, load_config
from src.constants import Move
from src.llm_player import LLMPlayer, LLMUnavailableError
from src.opponents import OPPONENTS, make_opponent, resolve_key
from src.prompts import MatchContext, PromptLibrary, build_reflection_prompt, build_turn_prompt
from src.reporting.tournament_report import TournamentReporter
from src.scorecard import (
    build_scorecard, compare_tiers_markdown, print_scorecard, records_from_tournament, write_scorecard,
)
from src.training_battery import TrainingBattery
from src.tournament import RoundRobinTournament
from src.usage_tracker import DailyLimitReachedException

LLM_NAMES = ("llmplayer", "llm_player", "llmlearner", "llmagent")


def parse_args():
    parser = argparse.ArgumentParser(description="LLM Strategy Learner V2")
    parser.add_argument("--config", default="config.toml", help="Path to the experiment configuration (TOML)")
    parser.add_argument("--output-dir", default=None, help="Directory for outputs (default: experiment.output_dir)")
    parser.add_argument("--tier", default=None, help="Prompt tier from prompts.toml (e.g. t1_informed)")
    parser.add_argument("--tiers", default=None,
                        help="Comma-separated tiers, or 'all': run the full pipeline per tier and compare them")
    parser.add_argument("--provider", choices=["openrouter", "gemini", "ollama", "mock"], default=None)
    parser.add_argument("--model", default=None, help="Override the model name from .env")
    parser.add_argument("--seed", type=int, default=None, help="Override experiment.seed (match lengths, Random opponent)")
    parser.add_argument("--quick", action="store_true",
                        help="Smoke test: 1 training match per strategy and 1 match per pairing")
    parser.add_argument("--train-only", action="store_true", help="Run only Phase 1 (training + Dreamer)")
    parser.add_argument("--eval-only", action="store_true", help="Run only Phase 2 (tournament), reusing saved playbook/memory")
    parser.add_argument("--no-llm", action="store_true", help="Tournament of classical strategies only (no API cost)")
    parser.add_argument("--show-prompt", default=None, metavar="TIER",
                        help="Print the prompts a tier produces for a sample match, then exit (no LLM calls)")
    parser.add_argument("--opponent", default="TitForTat", help="Opponent used by --show-prompt")
    return parser.parse_args()


def show_prompt(lib: PromptLibrary, tier: str, opponent: str, config: MasterConfig):
    key = resolve_key(opponent)
    ctx = MatchContext(opponent_key=key, total_rounds=10, avg_rounds=10.0, known_strategies=config.training.strategies)
    history = [(Move.COOPERATE, Move.DEFECT), (Move.DEFECT, Move.COOPERATE), (Move.COOPERATE, Move.COOPERATE)]
    spec = lib.tier(tier)
    print(f"##### TIER {tier}: {spec.description}")
    print(f"##### (sample: opponent {OPPONENTS[key].prompt_name}, round 4 of 10)\n")
    print(build_turn_prompt(lib, tier, ctx, history, 8, 5, ["[reflection] Example remembered lesson."],
                            ["Example note."], playbook=None))
    print("\n\n##### POST-MATCH REFLECTION PROMPT\n")
    print(build_reflection_prompt(lib, tier, ctx, history, 8, 5, [], ["[reflection] Example remembered lesson."]))


def build_llm(config: MasterConfig, lib: PromptLibrary, tier: str) -> LLMPlayer:
    return LLMPlayer(
        name="LLM Learner",
        provider=config.llm.provider,
        model_name=config.llm.model_name,
        tier=tier,
        prompt_library=lib,
        known_strategies=config.training.strategies,
        temperature=config.llm.temperature,
        max_rpm=config.llm.rate_limit.max_rpm,
        max_rpd=config.llm.rate_limit.max_rpd,
        max_retries=config.llm.max_retries,
        max_consecutive_fallbacks=config.llm.max_consecutive_fallbacks,
        memory_top_k=config.prompts.memory_top_k,
    )


def run_pipeline(config: MasterConfig, lib: PromptLibrary, tier: str, output_dir: str, args) -> dict:
    """Runs training + Dreamer + tournament for one tier. Returns {'training': card, 'tournament': card}."""
    cards: dict = {}
    llm_agent = None
    if not args.no_llm:
        llm_agent = build_llm(config, lib, tier)
        llm_agent.configure_output(output_dir)
        print(f"Provider: {llm_agent.provider} | Model: {llm_agent.model_name} | Tier: {tier}")

    if llm_agent is not None and not args.eval_only:
        battery = TrainingBattery(
            llm_player=llm_agent,
            strategies=config.training.strategies,
            matches_per_strategy=1 if args.quick else config.training.matches_per_strategy,
            rounds_per_match=config.training.rounds_per_match,
            error_rate=config.environment.error_rate,
            dreamer_enabled=config.training.dreamer.enabled,
            dreamer_temperature=config.training.dreamer.temperature,
            seed=config.experiment.seed,
        )
        battery.run_training(output_dir=output_dir)
        scorecard_path = os.path.join(output_dir, "scorecard_training.json")
        if os.path.exists(scorecard_path):
            with open(scorecard_path, encoding="utf-8") as f:
                cards["training"] = json.load(f)
    elif llm_agent is not None and args.eval_only:
        playbook_file = os.path.join(output_dir, "master_strategy_playbook.md")
        if os.path.exists(playbook_file):
            with open(playbook_file, encoding="utf-8") as f:
                llm_agent.set_master_playbook(f.read())
            print(f"Loaded playbook from {playbook_file}")
        if llm_agent.load_memory(os.path.join(output_dir, "memory.json")):
            print("Loaded memory from memory.json")
        llm_agent.freeze_notes = True

    if args.train_only:
        print("\nTraining-only mode complete. Skipping the tournament.")
        return cards

    if config.evaluation.enabled:
        players = []
        for name in config.evaluation.participants:
            if name.lower() in LLM_NAMES:
                if llm_agent is not None:
                    llm_agent.freeze_notes = True
                    players.append(llm_agent)
            else:
                players.append(make_opponent(name))

        tournament = RoundRobinTournament(
            players=players,
            matches_per_pairing=1 if args.quick else config.evaluation.matches_per_pairing,
            rounds_per_match=config.evaluation.rounds_per_match,
            error_rate=config.environment.error_rate,
            scoring_config=config.evaluation.scoring,
            seed=config.experiment.seed,
        )
        results = tournament.run_tournament(output_dir=output_dir)

        reporter = TournamentReporter(tournament_data=results, output_dir=output_dir)
        reporter.display_rich_console()
        if config.reporting.export_json:
            reporter.export_json()
        if config.reporting.export_markdown:
            reporter.export_markdown()
        if config.reporting.export_html:
            reporter.export_html()

        if llm_agent is not None:
            records = records_from_tournament(results["all_matches"], llm_agent.name)
            card = build_scorecard(
                records, results["llm_stats"].get(llm_agent.name, {}), tier=tier, phase="tournament",
                reveals_opponent=llm_agent.reveals_opponent)
            write_scorecard(card, output_dir, "scorecard_tournament")
            print_scorecard(card)
            cards["tournament"] = card

        checkpoint = RoundRobinTournament._checkpoint_path(output_dir)
        if os.path.exists(checkpoint):
            os.remove(checkpoint)
        print(f"\nTournament complete. All outputs saved to '{output_dir}'.")
    return cards


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.tier:
        config.prompts.tier = args.tier
    if args.provider:
        config.llm.provider = args.provider
    if args.model:
        config.llm.model_name = args.model
    if args.output_dir:
        config.experiment.output_dir = args.output_dir
    if args.seed is not None:
        config.experiment.seed = args.seed

    lib = PromptLibrary.load(config.prompts.prompts_file)

    if args.show_prompt:
        show_prompt(lib, args.show_prompt, args.opponent, config)
        return

    tiers = [config.prompts.tier]
    if args.tiers:
        tiers = list(lib.tiers) if args.tiers == "all" else [t.strip() for t in args.tiers.split(",")]
    for t in tiers:
        lib.tier(t)  # validate before spending anything
    if args.tiers and args.no_llm:
        sys.exit("--tiers needs an LLM; remove --no-llm.")

    base_dir = config.experiment.output_dir
    comparison = {}
    try:
        for tier in tiers:
            out_dir = os.path.join(base_dir, tier) if args.tiers else base_dir
            print(f"\n{'#' * 70}\n# TIER {tier}: {lib.tier(tier).description}\n{'#' * 70}")
            cards = run_pipeline(config, lib, tier, out_dir, args)
            phase = "tournament" if "tournament" in cards else "training"
            if phase in cards:
                comparison[tier] = cards[phase]
    except (DailyLimitReachedException, LLMUnavailableError) as e:
        print(f"\nRUN STOPPED: {e}\nProgress was checkpointed; run the same command again to resume.")
        sys.exit(2)

    if args.tiers and comparison:
        os.makedirs(base_dir, exist_ok=True)
        with open(os.path.join(base_dir, "tier_comparison.md"), "w", encoding="utf-8") as f:
            f.write(compare_tiers_markdown(comparison))
        with open(os.path.join(base_dir, "tier_comparison.json"), "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)
        print("\n" + compare_tiers_markdown(comparison))


if __name__ == "__main__":
    main()
