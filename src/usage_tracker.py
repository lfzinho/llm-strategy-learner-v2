import json
import os
import time
from datetime import datetime
from typing import Optional

from rich.console import Console


class DailyLimitReachedException(Exception):
    pass


class UsageTracker:
    """Enforces requests-per-minute / per-day limits and records token usage.

    Log format: {"requests": [unix timestamps], "prompt_tokens": n, "completion_tokens": n}.
    The older format (a bare list of timestamps) is still readable.
    """

    def __init__(self, log_file: Optional[str] = None, max_rpm=30, max_rpd=1500,
                 console: Optional[Console] = None, enabled: bool = True):
        self.log_file = log_file
        self.max_rpm = max_rpm
        self.max_rpd = max_rpd
        self.console = console
        self.enabled = enabled
        self.requests = []
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.quiet = False  # suppress the per-request usage line
        self._load_log()

    def set_log_file(self, log_file: str):
        self.log_file = log_file
        self._load_log()

    def _load_log(self):
        if not (self.enabled and self.log_file and os.path.exists(self.log_file)):
            return
        try:
            with open(self.log_file, "r") as f:
                data = json.load(f)
            stamps = data["requests"] if isinstance(data, dict) else data
            self.prompt_tokens = data.get("prompt_tokens", 0) if isinstance(data, dict) else 0
            self.completion_tokens = data.get("completion_tokens", 0) if isinstance(data, dict) else 0
            today = datetime.now().date()
            self.requests = [r for r in stamps if datetime.fromtimestamp(r).date() == today]
        except Exception as e:
            self._print(f"Warning: Could not load usage log {self.log_file}: {e}")
            self.requests = []

    def _save_log(self):
        if not self.log_file:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.log_file)), exist_ok=True)
        with open(self.log_file, "w") as f:
            json.dump({
                "requests": self.requests,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
            }, f)

    def get_rpm(self):
        now = time.time()
        return len([req for req in self.requests if now - req < 60])

    def get_rpd(self):
        return len(self.requests)

    def set_console(self, console: Console):
        self.console = console

    def _print(self, message: str):
        if self.console is not None:
            self.console.print(message)
        else:
            print(message)

    def record_tokens(self, prompt_tokens: int = 0, completion_tokens: int = 0):
        self.prompt_tokens += prompt_tokens or 0
        self.completion_tokens += completion_tokens or 0
        if self.enabled:
            self._save_log()

    def wait_and_record(self):
        """Enforces RPM limits and records the request."""
        if not self.enabled:
            return

        now = time.time()

        if self.get_rpd() >= self.max_rpd:
            self._print(f"[Tracker] CRITICAL: You have reached the daily limit of {self.max_rpd} RPD.")
            raise DailyLimitReachedException(f"Daily limit of {self.max_rpd} reached.")

        recent = [req for req in self.requests if now - req < 60]
        if len(recent) >= self.max_rpm:
            sleep_time = 60.0 - (now - min(recent))
            if sleep_time > 0:
                self._print(f"[Tracker] RPM limit ({self.max_rpm}) reached. Sleeping for {sleep_time:.1f}s...")
                time.sleep(sleep_time)
                now = time.time()

        self.requests.append(now)
        self._save_log()

        if not self.quiet:
            self._print(f"[Tracker] Usage: {self.get_rpm()} RPM | {self.get_rpd()} RPD (Daily Limit: {self.max_rpd})")
