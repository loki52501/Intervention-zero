"""Claude Code hook: keep work from going backwards.

PreToolUse (Bash/PowerShell): block commands that discard history or files.
PostToolUse (Edit/Write): append one line per edited file to CHANGES.md.
Exit code 2 blocks the tool call and shows stderr to the agent.
"""
import datetime
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]

BLOCKED = [
    r"git\s+reset\s+--hard",
    r"git\s+checkout\s+--\s",
    r"git\s+checkout\s+\.",
    r"git\s+restore\b",
    r"git\s+clean\s+-\w*f",
    r"git\s+push\s+.*(--force|-f\b)",
    r"git\s+rebase\b",
    r"git\s+branch\s+-D",
    r"git\s+stash\s+(drop|clear)",
    r"rm\s+-rf?\s+(iz|tests|scripts|\.git)\b",
    r"Remove-Item\s+.*-Recurse.*\b(iz|tests|scripts|\.git)\b",
    r"git\s+add\s+.*\.env\b(?!\.example)",
]


def main() -> None:
    data = json.load(sys.stdin)
    event = data.get("hook_event_name", "")
    tool = data.get("tool_name", "")
    inp = data.get("tool_input", {}) or {}

    if event == "PreToolUse" and tool in ("Bash", "PowerShell"):
        cmd = inp.get("command", "")
        for pat in BLOCKED:
            if re.search(pat, cmd, re.IGNORECASE):
                print(
                    f"Blocked by .claude/hooks/guard.py: '{pat}'. Changes must not go back; "
                    "make a new commit instead.",
                    file=sys.stderr,
                )
                sys.exit(2)

    if event == "PostToolUse" and tool in ("Edit", "Write", "MultiEdit"):
        path = inp.get("file_path", "")
        try:
            rel = pathlib.Path(path).resolve().relative_to(ROOT).as_posix()
        except ValueError:
            return
        if rel == "CHANGES.md" or rel.startswith(".git/"):
            return
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        with open(ROOT / "CHANGES.md", "a", encoding="utf-8") as f:
            f.write(f"- {stamp} {tool.lower()} `{rel}`\n")


if __name__ == "__main__":
    main()
