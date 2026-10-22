import time
import json
import os
from datetime import datetime
from typing import Optional

from rich.console import Console

class DailyLimitReachedException(Exception):
    pass

class UsageTracker:
    def __init__(self, log_file="output/api_usage.json", max_rpm=30, max_rpd=1500, console: Optional[Console] = None, enabled: bool = True):
        self.log_file = log_file
        self.max_rpm = max_rpm
        self.max_rpd = max_rpd
        self.console = console
        self.enabled = enabled
        self.requests = []
        if self.enabled:
            self._load_log()
        
    def _load_log(self):
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, "r") as f:
                    data = json.load(f)
                    # Filter for today's requests only (local time approximation)
                    today = datetime.now().date()
                    self.requests = [
                        req for req in data 
                        if datetime.fromtimestamp(req).date() == today
                    ]
            except Exception as e:
                print(f"Warning: Could not load usage log: {e}")
                self.requests = []
    
    def _save_log(self):
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        with open(self.log_file, "w") as f:
            json.dump(self.requests, f)
            
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

    def wait_and_record(self):
        """
        Enforces RPM limits and records the request.
        """
        if not self.enabled:
            return

        now = time.time()
        
        # Check Daily limit (RPD)
        if self.get_rpd() >= self.max_rpd:
            self._print(f"[Tracker] CRITICAL: You have reached the daily limit of {self.max_rpd} RPD.")
            raise DailyLimitReachedException(f"Daily limit of {self.max_rpd} reached.")
            
        # Check Minute limit (RPM)
        recent = [req for req in self.requests if now - req < 60]
        if len(recent) >= self.max_rpm:
            # Need to sleep until the oldest request in the 60s window falls out
            oldest_in_window = min(recent)
            sleep_time = 60.0 - (now - oldest_in_window)
            if sleep_time > 0:
                self._print(f"[Tracker] RPM limit ({self.max_rpm}) reached. Sleeping for {sleep_time:.1f}s...")
                time.sleep(sleep_time)
                now = time.time() # Update 'now' after sleeping
                
        # Record the new request
        self.requests.append(now)
        self._save_log()
        
        self._print(f"[Tracker] Usage: {self.get_rpm()} RPM | {self.get_rpd()} RPD (Daily Limit: {self.max_rpd})")
