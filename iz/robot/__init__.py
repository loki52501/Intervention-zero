"""Robot bodies for Intervention Zero (Unitree G1 from MuJoCo Menagerie, or a capsule fallback)."""
import os
from pathlib import Path

G1_DIR = Path(__file__).resolve().parents[2] / "assets" / "menagerie" / "unitree_g1"


def g1_available() -> bool:
    return (G1_DIR / "g1.xml").exists()


def robot_body() -> str:
    v = os.environ.get("ROBOT_BODY", "").strip().lower()
    if v in ("g1", "capsule"):
        return v
    return "g1" if g1_available() else "capsule"
