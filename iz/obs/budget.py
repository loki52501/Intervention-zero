"""LLM spend tracker."""
from __future__ import annotations

import logging
import os
import threading

log = logging.getLogger("iz.obs.budget")


class BudgetExceeded(Exception):
    pass


class Budget:
    def __init__(self, max_usd: float | None = None):
        self.max_usd = float(max_usd if max_usd is not None else (os.getenv("MAX_LLM_USD") or 15))
        self._lock = threading.Lock()
        self._total = 0.0
        self._by_model: dict[str, float] = {}
        self._warned = False

    def add(self, cost_usd: float, model: str = "") -> None:
        with self._lock:
            self._total += cost_usd
            self._by_model[model] = self._by_model.get(model, 0.0) + cost_usd
            total = self._total
            if not self._warned and total >= 0.8 * self.max_usd:
                self._warned = True
                log.warning("LLM budget at %.0f%% ($%.2f of $%.2f)", 100 * total / self.max_usd, total, self.max_usd)
        if total > self.max_usd:
            raise BudgetExceeded(f"LLM spend ${total:.2f} exceeds max ${self.max_usd:.2f}")

    @property
    def spent(self) -> float:
        return self._total

    @property
    def remaining(self) -> float:
        return max(0.0, self.max_usd - self._total)

    @property
    def fraction(self) -> float:
        return self._total / self.max_usd if self.max_usd else 1.0

    def snapshot(self) -> dict:
        with self._lock:
            return {"max_usd": self.max_usd, "spent": self._total,
                    "remaining": max(0.0, self.max_usd - self._total),
                    "fraction": self._total / self.max_usd if self.max_usd else 1.0,
                    "by_model": dict(self._by_model)}


_budget: Budget | None = None
_block = threading.Lock()


def get_budget() -> Budget:
    global _budget
    with _block:
        if _budget is None:
            _budget = Budget()
        return _budget
