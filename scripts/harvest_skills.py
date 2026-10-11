"""Run skill harvest (90s cap), write data/skill_sources.json, print counts (never keys)."""
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from iz.skills import harvest  # noqa: E402


def main() -> None:
    recs = harvest.harvest(time_cap=90.0)
    print("total", len(recs))
    print("by source", dict(collections.Counter(r["source"] for r in recs)))
    print("by platform", dict(collections.Counter(r["platform"] for r in recs)))
    if not recs:
        print("no records: no keys/cache; nothing invented")


if __name__ == "__main__":
    main()
