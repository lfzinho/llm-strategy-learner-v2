import argparse
import os
import sys
import io
from dotenv import load_dotenv

# Ensure Windows stdout handles UTF-8 (emojis, unicode tables)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure the simulation can find modules in the src/ directory
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

# Load environment variables
load_dotenv()

from src.config import load_config
from src.player_classes import get_strategy_class, STRATEGY_REGISTRY
from src.training_battery import TrainingBattery
from src.tournament import RoundRobinTournament
from src.reporting.tournament_report import TournamentReporter


def parse_args():
    parser = argparse.ArgumentParser(description="LLM Strategy Learner V2")
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to global configuration JSON file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["explicit_strategies", "generic_archetypes", "blind_optimization"],
        default=None,
        help="Override prompt mode (explicit_strategies, generic_archetypes, or blind_optimization)",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Run only Phase 1 (Training) & Dreamer synthesis",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run only Phase 2 (Championship Tournament)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Run tournament exclusively with classical strategies (zero API cost / dry run)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    # CLI overrides
    if args.mode:
        config.prompts.prompt_mode = args.mode

    try:
        from rich.console import Console
        from rich.panel import Panel
        console = Console()
        console.print(
            Panel.fit(
                f"[bold cyan]🎯 LLM Strategy Learner V2[/bold cyan]\n"
                f"[dim]Opponent Archetype Discovery & Round-Robin Championship[/dim]\n"
                f"[yellow]Prompt Mode:[/yellow] [bold]{config.prompts.prompt_mode}[/bold] | "
                f"[green]Noise:[/green] {config.environment.error_rate * 100:.0f}%",
                border_style="cyan",
            )
        )
    except Exception:
        print(f"=== LLM Strategy Learner V2 (Mode: {config.prompts.prompt_mode}) ===")

    llm_agent = None
    if not args.no_llm:
        from src.llm_player import LLMPlayer
        llm_agent = LLMPlayer(
            name="LLM Learner",
            provider=config.llm.provider,
            model_name=config.llm.model_name,
            prompt_mode=config.prompts.prompt_mode,
            known_strategies=config.training.strategies,
            temperature=config.llm.temperature,
            max_rpm=config.llm.rate_limit.max_rpm,
            max_rpd=config.llm.rate_limit.max_rpd,
        )
        print(f"Provider: {llm_agent.provider} | Model: {llm_agent.model_name} | Mode: {llm_agent.prompt_mode}")

    # =========================================================================
    # Phase 1: Structured Sequential Block Training & Dreamer Phase
    # =========================================================================
    if config.training.enabled and not args.eval_only and llm_agent is not None:
        training_battery = TrainingBattery(
            llm_player=llm_agent,
            strategies=config.training.strategies,
            matches_per_strategy=config.training.matches_per_strategy,
            rounds_per_match=config.training.rounds_per_match,
            error_rate=config.environment.error_rate,
            dreamer_enabled=config.training.dreamer.enabled,
            dreamer_temperature=config.training.dreamer.temperature,
        )
        training_history, master_playbook = training_battery.run_training(
            output_dir=config.experiment.output_dir
        )
    elif args.eval_only and llm_agent is not None:
        # Load existing master playbook if available
        playbook_file = os.path.join(config.experiment.output_dir, "master_strategy_playbook.md")
        if os.path.exists(playbook_file):
            with open(playbook_file, "r", encoding="utf-8") as f:
                saved_playbook = f.read()
            llm_agent.set_master_playbook(saved_playbook)
            print(f"Loaded existing Master Strategy Playbook from {playbook_file}")
        llm_agent.freeze_notes = True

    if args.train_only:
        print("\nTraining-only mode complete. Exiting before evaluation.")
        return

    # =========================================================================
    # Phase 2: Round-Robin Championship Tournament
    # =========================================================================
    if config.evaluation.enabled:
        tournament_players = []

        for p_name in config.evaluation.participants:
            if p_name.lower() in ("llmplayer", "llm_player", "llmlearner", "llmagent"):
                if llm_agent is not None:
                    # Freeze notes before tournament
                    llm_agent.freeze_notes = True
                    tournament_players.append(llm_agent)
            else:
                strat_cls = get_strategy_class(p_name)
                tournament_players.append(strat_cls())

        tournament = RoundRobinTournament(
            players=tournament_players,
            matches_per_pairing=config.evaluation.matches_per_pairing,
            rounds_per_match=config.evaluation.rounds_per_match,
            error_rate=config.environment.error_rate,
            scoring_config=config.evaluation.scoring,
        )

        tournament_results = tournament.run_tournament()

        # =========================================================================
        # Reporting: Rich TUI, JSON, Markdown, HTML Dashboard
        # =========================================================================
        reporter = TournamentReporter(
            tournament_data=tournament_results,
            output_dir=config.experiment.output_dir,
        )

        if config.reporting.terminal_style == "rich":
            reporter.display_rich_console()

        if config.reporting.export_json:
            reporter.export_json()

        if config.reporting.export_markdown:
            reporter.export_markdown()

        if config.reporting.export_html:
            reporter.export_html()

        print(f"\nTournament complete! All outputs saved to '{config.experiment.output_dir}' directory.")


if __name__ == "__main__":
    main()
