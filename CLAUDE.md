# Intervention Zero: agent rules

- Read `memory.yaml` before starting work; update your milestone status and append to its `log` when you finish.
- Interfaces live in `iz/types.py`. Do not change them without the main orchestrator's approval.
- Python: `.venv\Scripts\python.exe`. Windows, PowerShell. No bash-only scripts.
- Multiprocessing workers must be top-level functions; entry points need `if __name__ == "__main__":`.
- Secrets live only in `.env`. Never print, log or commit a key. Load with python-dotenv.
- `RAINDROP_ENABLED=0` by default. Replays and gate evaluations never send Raindrop events.
- Tests first for judge, gate, validators. Run `.venv\Scripts\python.exe -m pytest -q` before reporting done.
- Honesty: the world is a simulator we authored. Never invent numbers in README or output.
- Destructive git commands (reset --hard, checkout --, clean -f, push --force, restore) are blocked by a hook. Make a new commit instead.
- Commit after each milestone with a clear message.
