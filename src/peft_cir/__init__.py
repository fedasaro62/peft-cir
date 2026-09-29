from pathlib import Path

# repo root, so every default checkpoint / data / output path is written once and stays
# correct however deeply a module is nested
ROOT = Path(__file__).resolve().parents[2]
PRETRAINED = ROOT / "resources" / "pretrained"

__all__ = ["ROOT", "PRETRAINED"]
