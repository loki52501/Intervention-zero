"""Git pre-commit: refuse to commit .env files or anything that looks like a live secret."""
import re
import subprocess
import sys

staged = subprocess.run(
    ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
    capture_output=True, text=True, check=True,
).stdout.split()

bad = [f for f in staged if re.search(r"(^|/)\.env($|\.)", f) and not f.endswith(".env.example")]
if bad:
    print(f"pre-commit: refusing to commit env files: {bad}", file=sys.stderr)
    sys.exit(1)

SECRET = re.compile(
    r"(mongodb\+srv://[^\s\"']*:[^\s\"'@]+@"            # URI with password
    r"|sk-ant-[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9]{32,}"    # LLM keys
    r"|pa-[A-Za-z0-9_-]{30,}"                            # Voyage
    r"|\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b)"  # uuid-style keys
)
diff = subprocess.run(["git", "diff", "--cached", "-U0"], capture_output=True, text=True,
                      encoding="utf-8", errors="replace").stdout
hits = [l for l in diff.splitlines() if l.startswith("+") and SECRET.search(l)]
if hits:
    print(f"pre-commit: {len(hits)} added line(s) look like secrets; move them to .env", file=sys.stderr)
    sys.exit(1)
