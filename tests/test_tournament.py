import os
import tempfile
from src.opponents import make_opponent
from src.tournament import RoundRobinTournament
from src.reporting.tournament_report import TournamentReporter


def test_mini_round_robin_tournament():
    players = [
        make_opponent("AlwaysCooperate"),
        make_opponent("AlwaysDefect"),
        make_opponent("TitForTat"),
    ]

    tournament = RoundRobinTournament(
        players=players,
        matches_per_pairing=2,
        rounds_per_match=3,
        error_rate=0.0,
    )

    results = tournament.run_tournament()
    leaderboard = results["leaderboard"]
    assert len(leaderboard) == 3
    assert results["total_matches"] == 6  # 3 pairings * 2 matches each

    # Leaderboard ranks 1, 2, 3
    ranks = [p["rank"] for p in leaderboard]
    assert sorted(ranks) == [1, 2, 3]

    # Test reporting export
    with tempfile.TemporaryDirectory() as temp_dir:
        reporter = TournamentReporter(results, output_dir=temp_dir)
        json_path = reporter.export_json()
        md_path = reporter.export_markdown()
        html_path = reporter.export_html()

        assert os.path.exists(json_path)
        assert os.path.exists(md_path)
        assert os.path.exists(html_path)

        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()
            assert "Championship Tournament Leaderboard" in content
