#!/usr/bin/env python3
"""Link the files listed in docs/DATA_FILES.md into this checkout.

The analysis scripts read these files at repo-relative paths. Set NESI_DATA_ROOT
to a local copy of the data, then run this once; the scripts run unchanged.

    NESI_DATA_ROOT=/path/to/data python link_data.py          # size and SHA-256 checks
    python link_data.py --root /path/to/data --remove         # take the links out again

The linked paths are added to .git/info/exclude so they cannot be committed.
"""
import argparse
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
ROW = re.compile(r"^\| `([^`]+)` \| (\d+) \| `([0-9a-f]+)` \|")
EXCLUDE_START = "# link_data.py start"
EXCLUDE_END = "# link_data.py end"


def read_listing():
    rows = []
    for line in (REPO / "docs" / "DATA_FILES.md").read_text().splitlines():
        match = ROW.match(line)
        if match:
            rows.append((match[1], int(match[2]), match[3]))
    return rows


def resolve_root(arg):
    value = arg or os.environ.get("NESI_DATA_ROOT")
    if not value:
        sys.exit("Set NESI_DATA_ROOT or pass --root.")
    root = Path(value).expanduser().resolve()
    # reproduce/run_all.py keeps these files in a github_moved/ subfolder.
    if (root / "github_moved").is_dir():
        root = root / "github_moved"
    if not root.is_dir():
        sys.exit(f"Data folder not found: {root}")
    return root


def problem_with(source, size, sha_prefix):
    if not source.is_file():
        return "missing"
    if source.stat().st_size != size:
        return f"size {source.stat().st_size}, expected {size}"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if not digest.startswith(sha_prefix):
        return f"SHA-256 {digest[:12]}, expected {sha_prefix}"
    return None


def update_exclude(paths):
    out = subprocess.run(
        ["git", "rev-parse", "--git-path", "info/exclude"],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout.strip()
    exclude = (REPO / out).resolve()
    exclude.parent.mkdir(parents=True, exist_ok=True)
    text = exclude.read_text() if exclude.exists() else ""
    text = re.sub(rf"{EXCLUDE_START}\n.*?{EXCLUDE_END}\n", "", text, flags=re.S)
    if paths:
        block = "\n".join("/" + p for p in paths)
        text += f"{EXCLUDE_START}\n{block}\n{EXCLUDE_END}\n"
    exclude.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", help="local data folder (default: $NESI_DATA_ROOT)")
    parser.add_argument("--remove", action="store_true", help="remove links made by this script")
    args = parser.parse_args()
    rows = read_listing()

    if args.remove:
        removed = 0
        for path, _, _ in rows:
            target = REPO / path
            if target.is_symlink():
                target.unlink()
                removed += 1
        update_exclude([])
        print(f"Removed {removed} links.")
        return

    root = resolve_root(args.root)
    problems = [
        (path, issue)
        for path, size, sha_prefix in rows
        if (issue := problem_with(root / path, size, sha_prefix))
    ]
    for path, issue in problems:
        print(f"{path}: {issue}", file=sys.stderr)
    if problems:
        sys.exit(f"{len(problems)} of {len(rows)} files failed the check; nothing was linked.")

    blocked = [p for p, _, _ in rows if (REPO / p).exists() and not (REPO / p).is_symlink()]
    if blocked:
        sys.exit(f"Refusing to replace {len(blocked)} real files, for example {blocked[0]}.")

    update_exclude([path for path, _, _ in rows])
    for path, _, _ in rows:
        target = REPO / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            target.unlink()
        target.symlink_to(root / path)
    print(f"Linked {len(rows)} files from {root}.")


if __name__ == "__main__":
    main()
