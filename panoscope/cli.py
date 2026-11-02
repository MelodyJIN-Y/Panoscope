"""The ``panoscope`` console script: ``run``, ``build``, ``explain``, ``doctor``.

Thin by design. Every verb delegates to code that already exists and is already
tested; this module only parses arguments and prints. Heavy imports (pandas,
streamlit, the pipeline) happen *inside* the verb that needs them, so ``panoscope
doctor`` still answers on an install that is missing the ``app`` or ``agent``
extras — which is exactly the install you are trying to diagnose.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from panoscope.data import layout
from panoscope.data.study import Study, StudyNotFound

# Env vars that hold a secret. ``doctor`` reports presence only and never the value
# or any prefix of it; a diagnostic that leaks a key is worse than no diagnostic.
_CREDENTIAL_ENV = ("ANTHROPIC_API_KEY", "NCBI_API_KEY", "NCBI_ADMIN_EMAIL")

_DEP_GROUPS: dict[str, tuple[str, ...]] = {
    "core":   ("pandas", "numpy", "pyarrow", "pyyaml"),
    "app":    ("streamlit", "plotly"),
    "agent":  ("anthropic", "mcp", "python-dotenv"),
    "export": ("python-docx", "fpdf2"),
}


def _resolve_study_id(explicit: Optional[str]) -> str:
    """An id from the flag, the environment, or the one study on disk. Else explain."""
    sid = explicit or layout.default_study_id()
    if sid:
        return sid
    available = layout.available_studies()
    if len(available) == 1:
        return available[0]
    raise SystemExit(
        "No study specified. Pass --study, or set "
        f"{layout.ENV_STUDY}. Available: {', '.join(available) or '(none found)'}"
    )


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def _cmd_run(args: argparse.Namespace) -> int:
    """Launch the Streamlit app as a subprocess.

    Streamlit owns the process: it re-executes the script on every rerun, so the app
    cannot be called in-process. ``sys.executable -m streamlit`` keeps it in this
    interpreter rather than whichever ``streamlit`` is first on PATH, which is the
    bug that makes the UI silently render against an older Streamlit.
    """
    app = Path(__file__).resolve().parent / "app.py"
    if not app.is_file():  # pragma: no cover - only reachable on a broken install
        raise SystemExit(f"App entrypoint missing: {app}")
    env = os.environ.copy()
    if args.study:
        env[layout.ENV_STUDY] = args.study
    cmd = [sys.executable, "-m", "streamlit", "run", str(app)]
    # Streamlit's own STREAMLIT_* env vars configure port/headless/etc; no passthrough.
    print(f"[panoscope] {' '.join(cmd)}", flush=True)
    try:
        return subprocess.call(cmd, env=env)
    except FileNotFoundError:  # pragma: no cover
        raise SystemExit("Streamlit is not installed. pip install 'panoscope[app]'")


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def _cmd_build(args: argparse.Namespace) -> int:
    """(Re)build a study tree. Deterministic unless --notes asks for the live stage."""
    from panoscope.pipeline.run import run as run_pipeline

    study_id = _resolve_study_id(args.study)
    out = run_pipeline(study_id, notes=args.notes)
    print(f"[panoscope] built {out}" + (" (+ live notes)" if args.notes else ""))
    return 0


# --------------------------------------------------------------------------- #
# explain
# --------------------------------------------------------------------------- #
def _cmd_explain(args: argparse.Namespace) -> int:
    """Render the stored interpretation for one unit.

    TRANSITIONAL. This reads the committed per-unit JSON under
    ``<study>/interp/clusters/<unit>.json``, which is what the pipeline writes and the
    UI reads today. It moves to ``InterpretationRecord`` (panoscope/types/record.py)
    once that lands; the rendering below is a projection of the same fields, so the
    swap is a change of source, not of output.
    """
    study = Study.open(_resolve_study_id(args.study))
    unit = args.unit
    path = study.interp / "clusters" / f"{unit}.json"
    if not path.is_file():
        present = sorted(p.stem for p in (study.interp / "clusters").glob("*.json"))
        raise SystemExit(
            f"No interpretation for unit {unit!r} in study {study.id!r} ({path}). "
            f"Units present: {', '.join(present) or '(none)'}"
        )
    rec = json.loads(path.read_text(encoding="utf-8"))
    if args.json:
        print(json.dumps(rec, indent=2, ensure_ascii=False))
        return 0
    for line in _render_unit(study, rec):
        print(line)
    return 0


def _render_unit(study: Study, rec: dict[str, Any]) -> list[str]:
    """Pure projection of a stored unit record. No statistic is derived here."""
    out: list[str] = []
    unit = rec.get("cluster", "?")
    out.append(f"{study.id} / {unit} — {rec.get('cell_type', 'Unknown')}")
    out.append("=" * len(out[0]))
    band = rec.get("confidence", "?")
    score = rec.get("confidence_score")
    out.append(f"band          {band}" + (f" ({score})" if score is not None else ""))
    out.append(f"verify        {str(bool(rec.get('verify'))).upper()}")
    out.append(f"band basis    {rec.get('band_basis', '(not recorded)')}")
    out.append(f"lineage       {rec.get('lineage', '?')} / {rec.get('category', '?')}")
    if rec.get("small_n"):
        out.append("flag          small_n: few markers support this unit")

    out.append("")
    out.append("evidence (measured — jazzPanda output)")
    evidence = rec.get("evidence") or []
    if not evidence:
        out.append("  (none recorded)")
    for e in evidence:
        marks = "".join(
            (
                "*" if e.get("is_canonical") else " ",
                "!" if e.get("role") == "against" else " ",
            )
        )
        out.append(
            f"  {marks} {e.get('gene', '?'):<10} glm_coef={_num(e.get('glm_coef'))}"
            f"  pearson={_num(e.get('pearson'))}  top={e.get('top_cluster', '?')}"
            f"  role={e.get('role', '?')}"
        )
        for caveat in e.get("caveats") or []:
            out.append(f"      caveat: {caveat}")
    out.append("  (* canonical, ! argues against)")

    offpanel = rec.get("offpanel_notes") or []
    out.append("")
    out.append(f"never measured ({len(offpanel)}) — absence is NOT evidence")
    for n in offpanel:
        out.append(f"  - {n.get('gene', '?')}: {n.get('message', '')}")
    if not offpanel:
        out.append("  (none)")

    demotions = rec.get("demotions") or []
    if demotions:
        out.append("")
        out.append("band adjustments")
        out.extend(f"  - {d}" for d in demotions)

    if rec.get("notes"):
        out.append("")
        out.append("notes")
        out.append(f"  {rec['notes']}")

    trace = rec.get("source_trace") or []
    out.append("")
    out.append(f"source trace ({len(trace)})")
    out.extend(f"  {t}" for t in trace)
    out.append("")
    out.append(f"record        {study.interp / 'clusters' / (unit + '.json')}")
    return out


def _num(v: object) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) and not isinstance(v, bool) else "n/a"


# --------------------------------------------------------------------------- #
# doctor
# --------------------------------------------------------------------------- #
def _cmd_doctor(args: argparse.Namespace) -> int:
    """Resolved paths, studies, versions and credential presence. Never a secret."""
    print("panoscope doctor")
    print("================")
    print()
    print("interpreter")
    print(f"  python        {sys.version.split()[0]}  ({sys.executable})")
    print(f"  panoscope     {_dist_version('panoscope')}")
    print(f"  package at    {Path(__file__).resolve().parent}")

    print()
    print("paths")
    data_root = layout.resolve_data_root()
    print(f"  cwd           {Path.cwd()}")
    print(f"  data root     {data_root}  [{'ok' if data_root.is_dir() else 'MISSING'}]")
    print(f"  {layout.ENV_DATA_ROOT:<13} {os.getenv(layout.ENV_DATA_ROOT) or '(unset)'}")
    print(f"  {layout.ENV_STUDY:<13} {os.getenv(layout.ENV_STUDY) or '(unset)'}")
    print(f"  skills        {_pkg_dir_state('skills')}")
    print(f"  assets        {_pkg_dir_state('assets')}")

    print()
    studies = layout.available_studies()
    print(f"studies ({len(studies)})")
    if not studies:
        print("  (none found — no manifest.json under the study store)")
    for sid in studies:
        s = Study.open(sid)
        units = len(s.units)
        print(f"  {sid:<28} {units} units  {s.tissue or '?'} / {s.platform or '?'}")

    print()
    print("packages")
    for group, names in _DEP_GROUPS.items():
        rendered = "  ".join(f"{n}={_dist_version(n)}" for n in names)
        print(f"  {group:<7} {rendered}")

    print()
    print("credentials (presence only — never printed)")
    for name in _CREDENTIAL_ENV:
        print(f"  {name:<20} {'present' if os.getenv(name) else 'absent'}")
    dotenv = Path.cwd() / ".env"
    print(f"  {'.env in cwd':<20} {'present' if dotenv.is_file() else 'absent'}")
    return 0


def _dist_version(name: str) -> str:
    import importlib.metadata as md

    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return "not installed"


def _pkg_dir_state(name: str) -> str:
    d = Path(__file__).resolve().parent / name
    return f"{d}  [{'ok' if d.is_dir() else 'MISSING'}]"


# --------------------------------------------------------------------------- #
# entrypoint
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="panoscope",
        description="Panoscope: a human-in-the-loop interpretation layer for bioinformatics results.",
    )
    sub = ap.add_subparsers(dest="command", required=True, metavar="<command>")

    p_run = sub.add_parser("run", help="launch the review app")
    p_run.add_argument("--study", help=f"study id (default: ${layout.ENV_STUDY})")
    p_run.set_defaults(func=_cmd_run)

    p_build = sub.add_parser("build", help="(re)build a study tree (deterministic)")
    p_build.add_argument("--study", help="study id")
    p_build.add_argument(
        "--notes", action="store_true",
        help="also run the LIVE notes stage (real PubMed; needs network + credentials)",
    )
    p_build.set_defaults(func=_cmd_build)

    p_explain = sub.add_parser("explain", help="render the interpretation record for one unit")
    p_explain.add_argument("unit", help="unit id, e.g. c1")
    p_explain.add_argument("--study", help="study id")
    p_explain.add_argument("--json", action="store_true", help="emit the stored record verbatim")
    p_explain.set_defaults(func=_cmd_explain)

    p_doctor = sub.add_parser("doctor", help="resolved paths, studies, versions, credentials")
    p_doctor.set_defaults(func=_cmd_doctor)

    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except StudyNotFound as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
