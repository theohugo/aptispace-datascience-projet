from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

DOCS = {
    "readme": {
        "source": ROOT_DIR / "report" / "README.md",
        "target": ROOT_DIR / "README.md",
        "replacements": (
            ("](" + "assets/", "](" + "report/assets/"),
            ("](" + "../src/", "](" + "src/"),
        ),
    },
    "install": {
        "source": ROOT_DIR / "report" / "INSTALL.md",
        "target": ROOT_DIR / "INSTALL.md",
        "replacements": (),
    },
}


def sync_document(doc_name: str) -> int:
    config = DOCS.get(doc_name)
    if config is None:
        valid = ", ".join(sorted(DOCS))
        print(f"Unknown document '{doc_name}'. Expected one of: {valid}.")
        return 2

    source = config["source"]
    target = config["target"]

    if not source.exists():
        print(f"Source document not found: {source}")
        return 1

    text = source.read_text(encoding="utf-8")
    for old_text, new_text in config["replacements"]:
        text = text.replace(old_text, new_text)

    target.write_text(text, encoding="utf-8")
    source_label = source.relative_to(ROOT_DIR)
    target_label = target.relative_to(ROOT_DIR)
    print(f"Synced {source_label} -> {target_label}")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Usage: python tools/sync_project_docs.py <readme|install>")
        return 2

    return sync_document(argv[1].strip().lower())


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
