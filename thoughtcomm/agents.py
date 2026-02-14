from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

@dataclass
class Agent:
    role: str
    history: List[str] = field(default_factory=list)

    def reset(self):
        self.history.clear()

    def build_prompt(self, user_problem: str, round_idx: int, all_messages: List[str]) -> str:
        # Minimal, deterministic prompt format.
        parts = [f"System: {self.role}", f"User: {user_problem}"]
        if round_idx > 0 and all_messages:
            parts.append("Other agents' messages so far:")
            parts.extend(all_messages)
        parts.append("Assistant:")
        return "\n\n".join(parts)
