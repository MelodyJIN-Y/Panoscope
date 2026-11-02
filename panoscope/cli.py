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
from typing import Any, NamedTuple, Optional, Sequence

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
#: The interpretive question ``explain`` answers: *what IS this unit?* The id is a
#: registry key, validated by ``workflows.get`` rather than assumed -- the verb picks
#: a question, and the registry says which methods and which task kind answer it.
_WORKFLOW_ID = "marker-interpretation"

#: The entity column of a marker table. ``UnitSpec`` declares the *unit* column and
#: its reasoning but says nothing about which column names the entity, so this is the
#: one schema fact the verb still supplies. Named here rather than inline so the gap
#: is visible; it belongs on ``UnitSpec``, which is frozen.
_ENTITY_COLUMN = "gene"


def _cmd_explain(args: argparse.Namespace) -> int:
    """Render the interpretation record for one unit.

    The record is **built**, not read: ``panoscope explain`` goes through the same
    adapter, reader, method and ``core/`` functions the pipeline does, so every
    number on screen traces to the method's own table rather than to a JSON somebody
    could have edited. ``--json`` still emits the committed per-unit file verbatim,
    which is the v1 *compatibility view* (``core/compat.py``) and a different
    artifact from the record; ``--record`` emits the record itself.
    """
    study = Study.open(_resolve_study_id(args.study))
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

    built = _build_record(study, unit)
    if args.record:
        print(json.dumps(core_record.to_dict(built.record).value, indent=2,
                         ensure_ascii=False))
        return 0
    for line in _render_record(study, built):
        print(line)
    return 0


class _Built(NamedTuple):
    """One unit's record and the two things the record has no field for.

    ``InterpretationRecord`` carries the reading, the decisions, the claims and the
    undecidables -- it carries no band, no ``verify`` and no evidence items, so a
    renderer cannot print the current output from the record alone. Rather than
    pretend otherwise, the pieces the record was assembled FROM travel beside it and
    every line below says which of the two it came from.
    """

    record: Any          # InterpretationRecord
    evidence: Any        # NormalisedEvidence, glm_coef-descending
    verdict: Any         # core.verdict.Verdict
    canonical: frozenset[str]


def _build_record(study: Study, unit: str) -> _Built:
    """Study + unit -> ``InterpretationRecord``. Composition only; no statistic here.

    Heavy imports are local for the reason every verb's are: ``panoscope doctor``
    must answer on an install missing them.

    Two reads of the method's table, deliberately and not by accident. The tidy
    reader produces the ``ReadingRecord`` -- rows in, rows dropped and why, the
    entities no frame admits -- and the method module produces the evidence, because
    only it knows what its own specificity columns mean. Neither can produce the
    other's half today.
    """
    import dataclasses

    import pandas as pd

    from panoscope import methods, workflows
    from panoscope.core import measure, record as core_record
    from panoscope.core import verdict as core_verdict
    from panoscope.platforms.tidy import read_tidy
    from panoscope.platforms.xenium import XeniumAdapter
    from panoscope.types.measured import MeasuredWorld
    from panoscope.types.subject import EntityKind, gene, subject

    registered = methods.for_workflow(_WORKFLOW_ID)
    if len(registered) != 1:
        raise SystemExit(
            f"{len(registered)} methods answer {_WORKFLOW_ID!r} "
            f"({', '.join(m.name for m in registered) or 'none'}); explain cannot "
            "pick one for you. The record names its method, so build it with the "
            "pipeline and read it back."
        )
    method = registered[0]
    adapter = XeniumAdapter()          # no platform registry exists yet; see report
    path = study.inputs / "markers_top.csv"
    if not path.is_file():
        raise SystemExit(f"Study {study.id!r} has no method output at {path}.")

    # The unit column comes off the SCHEMA, through the method, never from here.
    spec = method.detect_unit(pd.read_csv(path))
    world = MeasuredWorld(adapter.frames(study))
    reading = read_tidy(
        path,
        unit_column=spec.unit_column,
        entity_column=_ENTITY_COLUMN,
        entity_kind=EntityKind.GENE,
        statistic_column=method.statistic_name,
        unit_kind=spec.unit_kind,
        world=world,
        clustering_mode=adapter.clustering_mode(study),
        # Whatever the method did not declare a unit is not a unit. Derived, so no
        # sentinel string ("NoSig") is restated outside the method that owns it.
        not_units=frozenset(str(v) for v in pd.read_csv(path)[spec.unit_column])
        - set(spec.units),
        unit_detection_because=spec.because,
        # Store-relative: an absolute path in a hashed record makes byte-stability a
        # property of whose laptop produced it.
        record_path=str(path.relative_to(study.dir)),
    ).value.record

    loaded = method.load(str(path), adapter, study)
    ev = loaded.value[unit]
    meta = study.unit_labels.get(unit, {})

    # Reference data, selecting which absences are worth reporting and nothing else:
    # the canonical markers this study's annotation lists for the label, and the ones
    # it records as off panel. The lattice decides what each one IS -- a gene named
    # here that the panel does measure comes back tested-absent, not never-measured.
    reported = frozenset().union(*(f.members for f in ev.reported)) if ev.reported \
        else frozenset()
    absences = tuple(
        measure.measure(
            measure.Reading(subject=subject("gene", g=gene(str(g))), statistic=None,
                            statistic_name=method.statistic_name),
            world, reported)
        for g in meta.get("offpanel_canonical", ())
    )

    # glm_coef descending, which is the order core/verdict.py says the CALLER owns:
    # the first eligible canonical marker is the driver, and the committed file is
    # not sorted within a cluster. See the report -- this belongs in the method.
    observed = tuple(sorted((i for i in ev.items if i.measured.carries_number),
                            key=lambda i: -float(i.statistic or 0.0)))
    ev = dataclasses.replace(ev, items=(
        observed
        + tuple(i for i in ev.items if not i.measured.carries_number)
        + tuple(t.value for t in absences)
    ))

    canonical = frozenset(str(g) for g in meta.get("canonical_markers", ()))
    verdict = core_verdict.assess(ev.items, canonical)
    record = core_record.build(
        ev, reading,
        core_record.unit_decisions(ev, loaded, *absences),
        verdict,
        task_kind=workflows.get(method.workflow).task_kind,
    ).value
    return _Built(record=record, evidence=ev, verdict=verdict.value,
                  canonical=frozenset(g.upper() for g in canonical))


def _band_from(record: Any) -> str:
    """The band, read back off the record's own spine.

    Every ``confidence``-step decision notes the band AFTER that step as its outcome
    (``core/verdict.py``), so the last one is the band and the ones before it are how
    it got there. Read rather than re-derived: a renderer that recomputed the band
    could disagree with the record it is rendering, which is the one disagreement a
    reader has no way to notice.
    """
    steps = [d for d in record.decisions if d.step == "confidence"]
    return steps[-1].outcome if steps else "(no band decision)"


def _render_record(study: Study, built: _Built) -> list[str]:
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
    out.append(f"  outside the capability frame: {len(r.entities_outside_frame)} "
               f"entity(s) the method named that no frame admits")

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

    never = [i for i in ev.items
             if i.measured.state is MeasuredState.NEVER_MEASURED]
    out.append("")
    out.append(f"never measured ({len(never)}) — absence is NOT evidence")
    for i in never:
        out.append(f"  - {i.subject.display()}: outside {i.measured.frame}")
    if not never:
        out.append("  (none among this label's canonical markers)")
    # The second never-measured fact, which is about the READ rather than the label:
    # a reader who saw only the line above would take "(0)" for "nothing was missed".
    out.append(f"  and {len(r.entities_outside_frame)} entity(s) in this read lie "
               "outside a capability frame — their absence argues nothing either")

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
    p_explain.add_argument("--json", action="store_true",
                           help="emit the committed v1 per-unit file verbatim")
    p_explain.add_argument("--record", action="store_true",
                           help="emit the InterpretationRecord as JSON")
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
