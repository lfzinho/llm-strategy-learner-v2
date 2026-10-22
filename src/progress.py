from typing import Optional

from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn


class ProgressTracker:
    """Tracks a global run progress and a current game/match progress in one Rich view."""

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=self.console,
            transient=False,
            expand=True,
        )
        self._started = False
        self.overall_task_id: Optional[int] = None
        self.current_task_id: Optional[int] = None
        self.status_task_id: Optional[int] = None

    def start(self):
        if not self._started:
            self.progress.start()
            self._started = True

    def stop(self):
        if self._started:
            self.progress.stop()
            self._started = False

    def add_overall_task(self, description: str, total: int) -> int:
        self.overall_task_id = self.progress.add_task(description, total=total)
        return self.overall_task_id

    def add_current_task(self, description: str, total: int) -> int:
        self.current_task_id = self.progress.add_task(description, total=total)
        return self.current_task_id

    def set_total(self, total: int, task: str = "overall"):
        task_id = self.overall_task_id if task == "overall" else self.current_task_id
        if task_id is None:
            return
        self.progress.update(task_id, total=total, completed=0)

    def update_overall(self, completed: int, total: Optional[int] = None):
        if self.overall_task_id is None:
            return
        if total is not None:
            self.progress.update(self.overall_task_id, total=total)
        self.progress.update(self.overall_task_id, completed=completed)

    def update_current(self, completed: int, total: Optional[int] = None):
        if self.current_task_id is None:
            return
        if total is not None:
            self.progress.update(self.current_task_id, total=total)
        self.progress.update(self.current_task_id, completed=completed)

    def set_status(self, text: str):
        """Shows a live one-line status (e.g. running cooperation rate) under the bars."""
        if self.status_task_id is None:
            self.status_task_id = self.progress.add_task(text, total=None, start=False)
        else:
            self.progress.update(self.status_task_id, description=text)

    def reset_current(self, description: str, total: int):
        if self.current_task_id is None:
            self.add_current_task(description, total=total)
            return
        self.progress.update(self.current_task_id, description=description, total=total, completed=0)
