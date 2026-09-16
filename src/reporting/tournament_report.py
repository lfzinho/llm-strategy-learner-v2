import json
import os
from typing import Any, Dict, List


class TournamentReporter:
    def __init__(self, tournament_data: Dict[str, Any], output_dir: str = "output"):
        self.data = tournament_data
        self.output_dir = output_dir
        self.leaderboard = tournament_data.get("leaderboard", [])
        self.player_names = tournament_data.get("player_names", [])
        self.cross_table = tournament_data.get("cross_table", {})
        self.all_matches = tournament_data.get("all_matches", [])

    def display_rich_console(self):
        try:
            from rich.console import Console
            from rich.table import Table
            from rich.panel import Panel
            console = Console()

            # Leaderboard Table
            lb_table = Table(
                title="🏆 Iterated Prisoner's Dilemma Championship Leaderboard",
                header_style="bold cyan",
                border_style="bright_blue",
            )
            lb_table.add_column("Rank", justify="center", style="bold yellow")
            lb_table.add_column("Player", style="bold white")
            lb_table.add_column("Strategy Archetype", style="dim")
            lb_table.add_column("MP", justify="center")
            lb_table.add_column("W", justify="center", style="green")
            lb_table.add_column("D", justify="center", style="yellow")
            lb_table.add_column("L", justify="center", style="red")
            lb_table.add_column("Match Pts", justify="right", style="bold magenta")
            lb_table.add_column("IPD Score", justify="right", style="bold cyan")
            lb_table.add_column("Avg/Match", justify="right")
            lb_table.add_column("Avg/Rnd", justify="right")

            for item in self.leaderboard:
                rank_str = f"#{item['rank']}"
                if item["rank"] == 1:
                    rank_str = "🥇 #1"
                elif item["rank"] == 2:
                    rank_str = "🥈 #2"
                elif item["rank"] == 3:
                    rank_str = "🥉 #3"

                lb_table.add_row(
                    rank_str,
                    item["name"],
                    item["strategy_name"],
                    str(item["matches_played"]),
                    str(item["wins"]),
                    str(item["draws"]),
                    str(item["losses"]),
                    f"{item['tournament_points']:.1f}",
                    str(item["total_ipd_score"]),
                    f"{item['avg_score_per_match']:.1f}",
                    f"{item['avg_score_per_round']:.2f}",
                )

            console.print()
            console.print(lb_table)
            console.print()

        except Exception as e:
            print(f"Terminal table rendering fallback: {e}")
            for item in self.leaderboard:
                print(f"Rank {item['rank']}: {item['name']} | Pts: {item['tournament_points']} | IPD Score: {item['total_ipd_score']}")

    def export_json(self) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        file_path = os.path.join(self.output_dir, "tournament_leaderboard.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)
        print(f"Exported JSON leaderboard: {file_path}")
        return file_path

    def export_markdown(self) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        file_path = os.path.join(self.output_dir, "tournament_leaderboard.md")

        md_lines = [
            "# 🏆 IPD Championship Tournament Leaderboard\n",
            f"**Total Matches**: {self.data.get('total_matches', 0)} | **Matches per Pairing**: {self.data.get('matches_per_pairing', 5)} | **Rounds per Match**: {self.data.get('rounds_per_match', 5)}\n",
            "| Rank | Player | Archetype | MP | W | D | L | Match Pts | IPD Score | Avg / Match | Avg / Round |",
            "| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]

        for item in self.leaderboard:
            rank_badge = f"**#{item['rank']}**"
            if item["rank"] == 1:
                rank_badge = "🥇 **1**"
            elif item["rank"] == 2:
                rank_badge = "🥈 **2**"
            elif item["rank"] == 3:
                rank_badge = "🥉 **3**"

            md_lines.append(
                f"| {rank_badge} | {item['name']} | {item['strategy_name']} | {item['matches_played']} | {item['wins']} | {item['draws']} | {item['losses']} | **{item['tournament_points']:.1f}** | **{item['total_ipd_score']}** | {item['avg_score_per_match']:.1f} | {item['avg_score_per_round']:.2f} |"
            )

        md_lines.append("\n---\n")
        md_lines.append("## ⚔️ Head-to-Head Cross Table (Scores & W-D-L)\n")

        # Head to head grid in markdown
        p_names = self.player_names
        header = "| Player | " + " | ".join(p_names) + " |"
        separator = "| :--- | " + " | ".join([":---:"] * len(p_names)) + " |"
        md_lines.append(header)
        md_lines.append(separator)

        for p1 in p_names:
            row = [f"**{p1}**"]
            for p2 in p_names:
                if p1 == p2:
                    row.append("-")
                else:
                    key = f"{p1}_vs_{p2}"
                    h2h = self.cross_table.get(key, {})
                    w, d, l = h2h.get("wins", 0), h2h.get("draws", 0), h2h.get("losses", 0)
                    s1 = h2h.get("p1_total_score", 0)
                    s2 = h2h.get("p2_total_score", 0)
                    row.append(f"{w}-{d}-{l}<br>({s1}-{s2})")
            md_lines.append("| " + " | ".join(row) + " |")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines) + "\n")

        print(f"Exported Markdown leaderboard: {file_path}")
        return file_path

    def export_html(self) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        file_path = os.path.join(self.output_dir, "tournament_report.html")

        # Top 3 players
        first_place = self.leaderboard[0] if self.leaderboard else None
        highest_scorer = max(self.leaderboard, key=lambda x: x["total_ipd_score"]) if self.leaderboard else None

        # Build rows for leaderboard
        lb_rows = []
        for item in self.leaderboard:
            rank = item["rank"]
            rank_class = f"rank-{rank}" if rank <= 3 else "rank-default"
            rank_icon = "🥇" if rank == 1 else ("🥈" if rank == 2 else ("🥉" if rank == 3 else f"#{rank}"))
            lb_rows.append(f"""
            <tr class="{rank_class}">
                <td class="rank-cell">{rank_icon}</td>
                <td class="name-cell"><strong>{item['name']}</strong></td>
                <td class="strategy-cell">{item['strategy_name']}</td>
                <td class="num-cell">{item['matches_played']}</td>
                <td class="num-cell win-cell">{item['wins']}</td>
                <td class="num-cell draw-cell">{item['draws']}</td>
                <td class="num-cell loss-cell">{item['losses']}</td>
                <td class="num-cell pts-cell"><strong>{item['tournament_points']:.1f}</strong></td>
                <td class="num-cell score-cell"><strong>{item['total_ipd_score']}</strong></td>
                <td class="num-cell">{item['avg_score_per_match']:.1f}</td>
                <td class="num-cell">{item['avg_score_per_round']:.2f}</td>
            </tr>
            """)
        lb_rows_html = "\n".join(lb_rows)

        # Build cross-table rows
        cross_headers = "".join([f"<th>{name}</th>" for name in self.player_names])
        cross_rows = []
        for p1 in self.player_names:
            cells = [f"<td class='row-player'><strong>{p1}</strong></td>"]
            for p2 in self.player_names:
                if p1 == p2:
                    cells.append("<td class='diagonal-cell'>—</td>")
                else:
                    key = f"{p1}_vs_{p2}"
                    h2h = self.cross_table.get(key, {})
                    w, d, l = h2h.get("wins", 0), h2h.get("draws", 0), h2h.get("losses", 0)
                    s1, s2 = h2h.get("p1_total_score", 0), h2h.get("p2_total_score", 0)
                    color_class = "cell-win" if w > l else ("cell-loss" if l > w else "cell-draw")
                    cells.append(f"<td class='{color_class}'><span class='record'>{w}W {d}D {l}L</span><span class='score'>{s1}:{s2}</span></td>")
            cross_rows.append(f"<tr>{''.join(cells)}</tr>")
        cross_rows_html = "\n".join(cross_rows)

        champion_name = first_place['name'] if first_place else 'N/A'
        champion_pts = first_place['tournament_points'] if first_place else 0
        scorer_name = highest_scorer['name'] if highest_scorer else 'N/A'
        scorer_pts = highest_scorer['total_ipd_score'] if highest_scorer else 0

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IPD Championship Tournament Report</title>
    <style>
        :root {{
            --bg-main: #0d1117;
            --bg-card: #161b22;
            --border-color: #30363d;
            --text-main: #f0f6fc;
            --text-dim: #8b949e;
            --accent-blue: #58a6ff;
            --accent-green: #3fb950;
            --accent-yellow: #d29922;
            --accent-red: #f85149;
            --accent-purple: #bc8cff;
        }}
        body {{
            background-color: var(--bg-main);
            color: var(--text-main);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 32px 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1, h2, h3 {{
            color: var(--text-main);
            font-weight: 600;
        }}
        .header-title {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 24px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 16px;
        }}
        .metric-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 16px;
            margin-bottom: 32px;
        }}
        .card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 20px;
        }}
        .card-label {{
            color: var(--text-dim);
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
        }}
        .card-value {{
            font-size: 1.6rem;
            font-weight: bold;
            color: var(--accent-blue);
        }}
        .table-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 32px;
            overflow-x: auto;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.95rem;
        }}
        th {{
            background: #21262d;
            color: var(--accent-blue);
            text-align: left;
            padding: 12px;
            border-bottom: 2px solid var(--border-color);
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid var(--border-color);
        }}
        .num-cell {{
            text-align: center;
        }}
        .rank-cell {{
            text-align: center;
            font-size: 1.1rem;
        }}
        .win-cell {{ color: var(--accent-green); }}
        .draw-cell {{ color: var(--accent-yellow); }}
        .loss-cell {{ color: var(--accent-red); }}
        .pts-cell {{ color: var(--accent-purple); font-size: 1.05rem; }}
        .score-cell {{ color: var(--accent-blue); font-size: 1.05rem; }}
        
        .rank-1 {{ background: rgba(210, 153, 34, 0.12); }}
        .rank-2 {{ background: rgba(139, 148, 158, 0.08); }}
        .rank-3 {{ background: rgba(187, 128, 9, 0.06); }}

        /* Cross table */
        .cross-table td, .cross-table th {{
            text-align: center;
            font-size: 0.85rem;
            padding: 8px;
        }}
        .row-player {{
            text-align: left !important;
            background: #21262d;
            white-space: nowrap;
        }}
        .diagonal-cell {{
            background: #0d1117;
            color: var(--text-dim);
        }}
        .cell-win {{
            background: rgba(63, 185, 80, 0.15);
            border-left: 2px solid var(--accent-green);
        }}
        .cell-draw {{
            background: rgba(210, 153, 34, 0.12);
        }}
        .cell-loss {{
            background: rgba(248, 81, 73, 0.12);
            border-left: 2px solid var(--accent-red);
        }}
        .record {{
            display: block;
            font-weight: bold;
        }}
        .score {{
            display: block;
            color: var(--text-dim);
            font-size: 0.75rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header-title">
            <h1>🎯 Iterated Prisoner's Dilemma Championship Tournament</h1>
        </div>

        <div class="metric-cards">
            <div class="card">
                <div class="card-label">Tournament Champion</div>
                <div class="card-value" style="color: var(--accent-yellow);">{champion_name}</div>
                <div style="font-size: 0.85rem; color: var(--text-dim); margin-top: 4px;">{champion_pts:.1f} Match Pts</div>
            </div>
            <div class="card">
                <div class="card-label">Highest IPD Payoff</div>
                <div class="card-value" style="color: var(--accent-green);">{scorer_name}</div>
                <div style="font-size: 0.85rem; color: var(--text-dim); margin-top: 4px;">{scorer_pts} Total Game Points</div>
            </div>
            <div class="card">
                <div class="card-label">Total Matches Completed</div>
                <div class="card-value">{self.data.get('total_matches', 0)}</div>
                <div style="font-size: 0.85rem; color: var(--text-dim); margin-top: 4px;">{self.data.get('matches_per_pairing', 5)} matches per pairing</div>
            </div>
        </div>

        <div class="table-card">
            <h2>🏆 Final Standings</h2>
            <table>
                <thead>
                    <tr>
                        <th style="text-align: center;">Rank</th>
                        <th>Player</th>
                        <th>Archetype</th>
                        <th style="text-align: center;">MP</th>
                        <th style="text-align: center;">W</th>
                        <th style="text-align: center;">D</th>
                        <th style="text-align: center;">L</th>
                        <th style="text-align: center;">Match Pts</th>
                        <th style="text-align: center;">IPD Score</th>
                        <th style="text-align: center;">Avg/Match</th>
                        <th style="text-align: center;">Avg/Rnd</th>
                    </tr>
                </thead>
                <tbody>
                    {lb_rows_html}
                </tbody>
            </table>
        </div>

        <div class="table-card">
            <h2>⚔️ Head-to-Head Cross-Table Matrix</h2>
            <p style="color: var(--text-dim); font-size: 0.9rem; margin-bottom: 12px;">
                Shows head-to-head match records (Wins - Draws - Losses) and cumulative IPD scores for each pairing.
            </p>
            <table class="cross-table">
                <thead>
                    <tr>
                        <th style="text-align: left;">Player vs Opponent</th>
                        {cross_headers}
                    </tr>
                </thead>
                <tbody>
                    {cross_rows_html}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"Exported HTML tournament report: {file_path}")
        return file_path
