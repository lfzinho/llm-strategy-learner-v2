"""Memory stream after Park et al. 2023 ("Generative Agents"), without embeddings.

Entries are observations (facts measured by the harness) or reflections (lessons
the model wrote, citing the observations they came from). Retrieval ranks entries
by recency + importance + relevance, so the prompt stays small and one repeated
lesson cannot crowd out everything else.
"""
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {"the", "a", "an", "and", "or", "of", "to", "in", "is", "it", "on", "if", "then", "you", "your",
              "they", "their", "with", "for", "as", "at", "by", "this", "that", "be", "are", "was", "will"}


def tokens(text: str) -> set:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS}


def similarity(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


@dataclass
class MemoryEntry:
    id: int
    kind: str  # "observation" | "reflection"
    text: str
    importance: int  # 1-10
    step: int  # match counter when it was created
    opponent: Optional[str] = None  # registry key, harness-known
    evidence: List[int] = field(default_factory=list)


class MemoryStream:
    def __init__(self, recency_decay: float = 0.85, dedupe_threshold: float = 0.7):
        self.entries: List[MemoryEntry] = []
        self.step = 0
        self.recency_decay = recency_decay
        self.dedupe_threshold = dedupe_threshold

    def advance(self):
        """Call once per finished match; older entries get less recent."""
        self.step += 1

    def add(self, kind: str, text: str, importance: int = 5, opponent: Optional[str] = None,
            evidence: Optional[List[int]] = None) -> Optional[MemoryEntry]:
        """Adds an entry. Returns None if a near-duplicate of the same kind exists."""
        text = text.strip()
        if not text:
            return None
        importance = max(1, min(10, int(importance)))
        for existing in self.entries:
            if existing.kind == kind and similarity(existing.text, text) >= self.dedupe_threshold:
                existing.importance = max(existing.importance, importance)  # restated lessons gain weight
                existing.step = self.step
                return None
        entry = MemoryEntry(
            id=len(self.entries) + 1, kind=kind, text=text, importance=importance,
            step=self.step, opponent=opponent, evidence=list(evidence or []),
        )
        self.entries.append(entry)
        return entry

    def retrieve(self, query: str = "", k: int = 8, opponent: Optional[str] = None) -> List[MemoryEntry]:
        """Top-k entries by recency + importance + relevance, each scaled to 0-1.

        `opponent` should only be passed in tiers that reveal the opponent; otherwise
        the tag would leak the identity.
        """
        q = tokens(query)

        def score(e: MemoryEntry) -> float:
            recency = self.recency_decay ** max(0, self.step - e.step)
            importance = e.importance / 10
            relevance = 0.0
            if opponent and e.opponent == opponent:
                relevance = 1.0
            elif q:
                et = tokens(e.text)
                relevance = len(q & et) / len(q | et) if et else 0.0
            return recency + importance + relevance

        ranked = sorted(self.entries, key=score, reverse=True)
        return ranked[:k]

    def reflections(self) -> List[MemoryEntry]:
        return [e for e in self.entries if e.kind == "reflection"]

    def render(self, entries: List[MemoryEntry]) -> List[str]:
        return [f"[{e.kind}] {e.text}" for e in entries]

    def to_list(self) -> List[Dict]:
        return [asdict(e) for e in self.entries]

    def load_list(self, data: List[Dict], step: int = 0):
        self.entries = [MemoryEntry(**d) for d in data]
        self.step = step
