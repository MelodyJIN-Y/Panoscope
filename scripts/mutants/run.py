"""Curated mutation run over the high-stakes core (TEST_PLAN §2.4). Never mutates the repo.

Copies the checkout ONCE to a temp dir and mutates only the copy: a crash mid-run can
leave a mutated ``core/`` behind, and a mutated build can write into ``data/studies``.
Each mutant in ``mutants.json`` is a single-occurrence string replacement plus the test
files that should catch it; a mutant marked ``equivalent`` is listed, not run.

    .venv/bin/python scripts/mutants/run.py              # every mutant
    .venv/bin/python scripts/mutants/run.py G3 V1        # just these
    .venv/bin/python scripts/mutants/run.py --out r.json # also write the rows

Exit status 1 when a non-equivalent mutant survives.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_SKIP = {".venv", ".git", ".claude", "__pycache__", ".pytest_cache", ".hypothesis", "node_modules"}


def _skip(d: str, names: list[str]) -> set[str]:
    return (_SKIP | ({"data"} if Path(d) == REPO else set())) & set(names)


def _copy(dest: Path) -> Path:
    """The repo without its heavy, never-written parts. ``data/studies`` is copied (tests
    write per-test copies of it, and the tree guard reads it); its ``viz/`` and every
    other ``data/`` entry are symlinked read-only inputs."""
    shutil.copytree(REPO, dest, ignore=_skip)
    (dest / "data").mkdir()
    for p in (REPO / "data").iterdir():
        if p.name == "studies":
            shutil.copytree(p, dest / "data" / "studies", ignore=shutil.ignore_patterns("viz"))
            for viz in p.glob("*/viz"):
                (dest / "data" / "studies" / viz.parent.name / "viz").symlink_to(viz)
        else:
            (dest / "data" / p.name).symlink_to(p)
    return dest


def main(argv: list[str]) -> int:
    out = argv[argv.index("--out") + 1] if "--out" in argv else None
    want = [a for a in argv if not a.startswith("--") and a != out]
    mutants = json.loads(Path(__file__).with_name("mutants.json").read_text())
    if want:
        mutants = [m for m in mutants if m["id"] in want]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    rows, survivors = [], []
    with tempfile.TemporaryDirectory(prefix="panoscope-mutants-") as tmp:
        work = _copy(Path(tmp).resolve() / "repo")
        env["PYTHONPATH"] = str(work)
        where = subprocess.run([sys.executable, "-c", "import panoscope; print(panoscope.__file__)"],
                               cwd=work, env=env, capture_output=True, text=True).stdout.strip()
        assert where.startswith(str(work)), f"the copy is not what gets imported: {where}"
        for m in mutants:
            if m.get("equivalent"):
                rows.append({**m, "status": "EQUIVALENT"})
                print(f"{m['id']:5} EQUIVALENT  {m['equivalent']}", flush=True)
                continue
            src = work / m["file"]
            text = src.read_text()
            if text.count(m["old"]) != 1:
                rows.append({**m, "status": f"STALE ({text.count(m['old'])} matches)"})
                survivors.append(m["id"])
                print(f"{m['id']:5} STALE       the mutated text is no longer in {m['file']}")
                continue
            t0 = time.time()
            try:
                src.write_text(text.replace(m["old"], m["new"]))
                r = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider",
                                    "-m", "not live and not browser", *m["tests"]],
                                   cwd=work, env=env, capture_output=True, text=True, timeout=1800)
            finally:
                src.write_text(text)
            killer = next((ln for ln in r.stdout.splitlines() if ln.startswith(("FAILED", "ERROR"))), "")
            # Only a FAILED test is a kill; an ERROR (collection, import) is a broken run.
            status = ("SURVIVED" if r.returncode == 0 else "KILLED" if killer.startswith("FAILED")
                      else f"ERROR rc={r.returncode}")
            # A killer that also fails on the unmutated copy was broken already, not killed.
            if status == "KILLED" and subprocess.run(
                    [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                     killer.removeprefix("FAILED ").split(" - ")[0]],
                    cwd=work, env=env, capture_output=True, timeout=1800).returncode != 0:
                status = "ERROR baseline"
            rows.append({**m, "status": status, "secs": round(time.time() - t0), "killer": killer})
            if status != "KILLED":
                survivors.append(m["id"])
            print(f"{m['id']:5} {status:11} {round(time.time() - t0):4}s  {killer}", flush=True)
    if out:
        Path(out).write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    print(f"\n{len(rows)} mutants: {sum(r['status'] == 'KILLED' for r in rows)} killed, "
          f"{sum(r['status'] == 'EQUIVALENT' for r in rows)} equivalent, "
          f"survivors: {', '.join(survivors) or 'none'}")
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
