"""The ``panoscope`` console script: ``run``, ``build``, ``explain``, ``doctor``, ``reference fetch``.

Thin: every verb delegates and this module only parses arguments and prints. Heavy
imports happen *inside* the verb that needs them, so ``panoscope doctor`` still
answers on an install missing the ``app`` or ``agent`` extras.
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
    # Before the pipeline, because ``run()`` creates the dataset directory as its
    # first act: an unknown id left an empty directory in the store and then died
    # with a raw traceback.
    Study.open(study_id).require()
    out = run_pipeline(study_id, notes=args.notes)
    print(f"[panoscope] built {out}" + (" (+ live notes)" if args.notes else ""))
    return 0


# --------------------------------------------------------------------------- #
# explain
# --------------------------------------------------------------------------- #
# The composition root ``explain`` renders lives in ``pipeline/records.py``, shared
# with ``panoscope build``.


def _cmd_explain(args: argparse.Namespace) -> int:
    """Render the interpretation record for one unit.

    The record is **built**, not read, through the same functions the pipeline uses.
    ``--json`` emits the committed v1 per-unit file (``core/compat.py``'s view);
    ``--record`` emits the record itself.
    """
    # ``require`` names the typo and lists the real studies. ``Study.open``
    # deliberately never validates, so a mistyped --study was reported as an
    # existing study with no units -- blaming the unit for a study-id error.
    study = Study.open(_resolve_study_id(args.study)).require()
    # Stage 0, the same gate ``build`` runs: a store missing a capability frame
    # must refuse, not render a complete-looking record without that frame.
    from panoscope.pipeline.stages.validate import validate

    try:
        validate(study.id)
    except ValueError as exc:  # a blocked store is a user error, not a traceback
        raise SystemExit(str(exc)) from exc
    unit = args.unit
    if unit not in study.units:
        raise SystemExit(
            f"No interpretation for unit {unit!r} in study {study.id!r}. "
            f"Units present: {', '.join(study.units) or '(none)'}"
        )

    if args.json:
        path = study.interp / "clusters" / f"{unit}.json"
        if not path.is_file():
            raise SystemExit(f"No committed v1 record at {path}.")
        print(json.dumps(json.loads(path.read_text(encoding="utf-8")),
                         indent=2, ensure_ascii=False))
        return 0

    from panoscope.core import record as core_record
    from panoscope.pipeline import records

    built = records.build(study, unit)
    if args.record:
        print(json.dumps(core_record.to_dict(built.record).value, indent=2,
                         ensure_ascii=False))
        return 0
    for line in _render_record(study, built):
        print(line)
    return 0


def _band_from(record: Any) -> str:
    """The band, read back off the record's own spine.

    The last ``confidence``-step decision's outcome (``core/verdict.py``); read, never
    re-derived, so the renderer cannot disagree with the record.
    """
    steps = [d for d in record.decisions if d.step == "confidence"]
    return steps[-1].outcome if steps else "(no band decision)"


def _render_record(study: Study, built: Any) -> list[str]:
    """Pure projection of one record. No statistic is derived here."""
    from panoscope.core import record as core_record
    from panoscope.types.measured import MeasuredState

    rec, ev, verdict = built.record, built.evidence, built.verdict
    meta = study.unit_labels.get(rec.unit, {})
    out: list[str] = [f"{study.id} / {rec.unit} — {meta.get('cell_type', 'Unknown')}"]
    out.append("=" * len(out[0]))
    out.append(f"band          {_band_from(rec):<16}[from the record's decisions]")
    out.append(f"verify        {str(bool(verdict.verify)).upper():<16}"
               "[core/verdict rubric; the record has no field for it]")
    out.append(f"band basis    {verdict.basis}")
    out.append(f"lineage       {meta.get('lineage', '?')} / {meta.get('category', '?')}")
    out.append(f"task          {rec.task_kind.value}  ·  {ev.module}  "
               f"·  spec {rec.spec_version}")
    generated = ", ".join(
        f"{f}={'present' if getattr(rec, f) is not None else 'not generated'}"
        for f in core_record.GENERATED_FIELDS
    )
    out.append(f"generated     {generated}  (every other field is deterministic)")
    if verdict.fragile:
        out.append("flag          small_n: few markers support this unit")

    r = rec.reading
    out.append("")
    out.append(f"reading       {r.path}")
    out.append(f"  sha256      {r.sha256}")
    out.append(f"  columns     unit={r.unit_column}  statistic={r.statistic_column}"
               f"  unit_kind={r.unit_kind}")
    out.append(f"  rows        {r.rows_in} read, {r.rows_dropped} dropped")
    out.extend(f"                - {why}: {n}" for why, n in r.drop_reasons)
    out.append(f"  clustering  {r.clustering_mode}")
    out.append("  frames      " + "  ".join(f"{f.name}={f.size}" for f in r.frames))
    # "at least one frame", not "no frame". ``read_tidy`` builds this set with
    # ``not all(f.contains(e) for f in kind_frames)``, and this study declares TWO
    # gene frames -- panel (280) and detect (231). The 49 no-signal genes land here
    # by failing detect alone, while the panel admits every one of them, so the old
    # wording asserted they were never probed. That is the collapse the four-state
    # lattice exists to stop, printed by the command a reviewer uses to audit it.
    out.append(f"  outside the capability frame: {len(r.entities_outside_frame)} "
               f"entity(s) the method named that at least one frame does not admit")

    out.append("")
    out.append(f"evidence (measured — {ev.module} {r.statistic_column})")
    observed = [i for i in ev.items if i.measured.carries_number]
    if not observed:
        out.append("  (none recorded)")
    for i in observed:
        canon = any(e.id.upper() in built.canonical for e in i.subject.entities())
        out.append(
            f"  {'*' if canon else ' '} {i.subject.display():<10} "
            f"{r.statistic_column}={_num(i.statistic)}"
            f"  {'  '.join(f'{k}={_num(v)}' for k, v in i.corroborating)}"
        )
        out.extend(f"      caveat: {f}" for f in i.flags)
    out.append("  (* canonical for this label, from the study annotation)")

    # The productive half, between the two this already printed. A record that lists
    # what was observed and what was never measured, and says nothing about what was
    # measured and came up empty, reads as though nothing else was tested -- the
    # four-states-into-two collapse, in the renderer rather than in the lattice.
    # The canonical ones are NAMED because the study's marker list selects which
    # absences are worth reporting; it supplies no value, and there is none to supply.
    tested = [i for i in ev.items
              if i.measured.state is MeasuredState.TESTED_ABSENT]
    out.append("")
    out.append(f"tested, not reported here ({len(tested)}) — absence IS evidence")
    canon_absent = [i for i in tested
                    if any(e.id.upper() in built.canonical
                           for e in i.subject.entities())]
    for i in canon_absent:
        out.append(f"  * {i.subject.display()}: in {i.measured.frame}, tested, "
                   "peaked elsewhere")
    if not tested:
        out.append("  (none — every entity in frame was reported for this unit)")
    elif not canon_absent:
        out.append("  (none of this label's canonical markers is among them)")

    never = [i for i in ev.items
             if i.measured.state is MeasuredState.NEVER_MEASURED]
    out.append("")
    out.append(f"never measured ({len(never)}) — absence is NOT evidence")
    for i in never:
        out.append(f"  - {i.subject.display()}: outside {i.measured.frame}")
    if not never:
        out.append("  (none among this label's canonical markers)")
    # The read-level fact, which is about the READ rather than this label's canon.
    # Only names the list above did not already print (``entities_outside_frame``
    # is a superset), so nothing is counted twice.
    shown = {i.subject.display() for i in never}
    extra = [e for e in r.entities_outside_frame if str(e) not in shown]
    if extra:
        out.append(f"  and {len(extra)} further entity(s) in this read lie outside "
                   "a capability frame — their absence argues nothing either")
    elif not never:
        out.append("  (and nothing in this read lies outside a capability frame)")

    if verdict.demotions:
        out.append("")
        out.append(f"band adjustments ({len(verdict.demotions)})")
        for d in rec.decisions:
            if d.step == "confidence" and d.rule in verdict.demotions:
                out.append(f"  - {d.rule} -> {d.outcome}: {'; '.join(d.because)}")

    if rec.claims_used:
        out.append("")
        out.append(f"claims used ({len(rec.claims_used)})")
        out.extend(f"  {cid} -> {v}" for cid, v in rec.claims_used)
    if rec.undecidable:
        out.append("")
        out.append(f"undecidable ({len(rec.undecidable)})")
        out.extend(f"  - {u}" for u in rec.undecidable)

    out.append("")
    out.append(f"source trace ({len(rec.decisions)}) — every line resolves to its "
               "subject")
    for d in rec.decisions:
        out.append(f"  {d.rule} -> {d.outcome}"
                   + (f"  [{'  '.join(d.refs)}]" if d.refs else ""))
    out.append("")
    out.append(f"record        {core_record.path_for(study.interp, rec).value}"
               "  (not written by explain)")
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
    how = "from env" if os.getenv(layout.ENV_DATA_ROOT) else f"default; set {layout.ENV_DATA_ROOT} to move"
    print(f"  user studies  {layout.user_data_root()}  (studies built from a results folder; {how})")
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
    from panoscope.reference import sources as ref

    print(f"reference sources (cache {ref.cache_root()})")
    for s in ref.fetchable():
        state = "cached" if s.path.is_file() else "not fetched"
        print(f"  {type(s).__name__:<13} {s.release:<12} [{state}]  licence: {s.licence}")

    print()
    print("packages")
    for group, names in _DEP_GROUPS.items():
        rendered = "  ".join(f"{n}={_dist_version(n)}" for n in names)
        print(f"  {group:<7} {rendered}")

    print()
    print("credentials (presence only — never printed)")
    # Load .env first, because the agent does: PanoscopeAgent.__init__ calls
    # load_dotenv() before reading ANTHROPIC_API_KEY, and mcp_client does the same
    # for the NCBI pair. Reporting bare os.getenv printed "absent" for all three
    # one line above doctor's own ".env in cwd: present" -- on a machine where the
    # app would go live. doctor is the documented way to check this, and the README
    # tells the reader to put exactly these keys in .env.
    try:
        from dotenv import load_dotenv

        # The cwd .env specifically, not a parent walk: doctor reports ".env in
        # cwd" one line below, so it must answer about that same file.
        load_dotenv(Path.cwd() / ".env")
    except Exception:  # noqa: BLE001 - no python-dotenv: report the shell as-is
        pass
    for name in _CREDENTIAL_ENV:
        print(f"  {name:<20} {'present' if os.getenv(name) else 'absent'}")
    dotenv = Path.cwd() / ".env"
    print(f"  {'.env in cwd':<20} {'present' if dotenv.is_file() else 'absent'}"
          f"  ({dotenv})")
    # What this answer is ABOUT, said rather than implied. doctor reads the shell
    # plus THIS file; the agent and the MCP client call bare ``load_dotenv()``,
    # which resolves from the installed package and walks up. Run from anywhere
    # but the repo the two can disagree, and a bare "absent" then sends the
    # reader hunting for a key the app will find. doctor does not adopt the
    # agent's resolution, because reading whatever .env happens to sit above the
    # cwd is how this command came to report the developer's own credentials.
    print("  (shell + the .env above; the agent resolves its own, "
          "relative to the installed package)")
    return 0


def _cmd_reference_fetch(args: argparse.Namespace) -> int:
    """Download every reference source into the user cache; one failure does not stop the rest."""
    from panoscope.reference import sources as ref

    failed = 0
    for s in ref.fetchable():
        try:
            print(f"{type(s).__name__} {s.release}: {s.fetch()}  (licence: {s.licence})")
        except ref.SourceUnavailable as exc:
            failed += 1
            print(f"{type(s).__name__} {s.release}: not fetched -- {exc}")
    return 1 if failed else 0


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
    p_explain.add_argument("--json", action="store_true",
                           help="emit the committed v1 per-unit file verbatim")
    p_explain.add_argument("--record", action="store_true",
                           help="emit the InterpretationRecord as JSON")
    p_explain.set_defaults(func=_cmd_explain)

    p_doctor = sub.add_parser("doctor", help="resolved paths, studies, versions, credentials")
    p_doctor.set_defaults(func=_cmd_doctor)

    p_ref = sub.add_parser("reference", help="reference sources (fetched, never bundled)")
    p_ref_sub = p_ref.add_subparsers(dest="ref_command", required=True, metavar="<action>")
    p_ref_sub.add_parser("fetch", help="download every source into the user cache"
                         ).set_defaults(func=_cmd_reference_fetch)

    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except StudyNotFound as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
