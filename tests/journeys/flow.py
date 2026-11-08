"""The biologist's steps through the app, shared by the journeys (AppTest)."""
from __future__ import annotations

import shutil
from pathlib import Path

from tests.ui.screen import APP, run

DEMO_INPUTS = Path(__file__).resolve().parents[2] / "data" / "studies" / "xenium_hbreast_sample1" / "inputs"


def folder_with(tmp: Path, name: str, files: dict[str, Path | str]) -> Path:
    """A collaborator's results folder: ``{name in folder: source path or text}``."""
    folder = tmp / name
    folder.mkdir(parents=True)
    for dst, src in files.items():
        (folder / dst).parent.mkdir(parents=True, exist_ok=True)
        if isinstance(src, Path):
            shutil.copyfile(src, folder / dst)
        else:
            (folder / dst).write_text(src)
    return folder


def upload():
    from streamlit.testing.v1 import AppTest

    # 90 s, not 300: a script thread killed by SystemExit hangs AppTest to the timeout.
    at = AppTest.from_file(APP, default_timeout=90)
    at.session_state["active_page"] = "upload"
    return run(at)


def paste(at, folder: Path, platform: str = "xenium"):
    at = run(at.text_input(key="ob_folder").input(str(folder)))
    if "ob_platform" not in {w.key for w in at.selectbox}:    # an existing study keeps its own
        return at
    return run(at.selectbox(key="ob_platform").select(platform))


def declare(at, **fields):
    for f, v in fields.items():
        at.text_input(key=f"ob_f_{f}").input(v)
    return run(at.button(key="ob_apply").click())


def build_and_start(at):
    at = run(at.button(key="ob_build").click())
    assert not at.error, [e.value for e in at.error]          # a refused build, named
    return run(at.button(key="ob_corpus_skip").click())


def onboard(folder: Path, platform: str = "xenium", at=None, **fields):
    """Path box -> platform -> declare -> Build -> Start annotation (in ``at``'s session)."""
    at = paste(go(at, "upload") if at is not None else upload(), folder, platform)
    asked = {w.key for w in at.text_input}
    fields = {f: v for f, v in fields.items() if f"ob_f_{f}" in asked}
    if fields:
        at = declare(at, **fields)
    return build_and_start(at)


def go(at, page: str, cluster: str | None = None):
    at.session_state["active_page"] = page
    if cluster:
        at.session_state["selected_cluster"] = cluster
    return run(at)


def call_it(at, cluster: str, *, pick: str | None = None, free: str | None = None,
            basis: str | None = None):
    """Markers page: 'Call it yourself' -> a candidate or own label -> Save call."""
    at = go(at, "markers", cluster)
    run(at.toggle(key=f"rv_calling_{cluster}").set_value(True))
    if pick:
        run(at.selectbox(key=f"rv_pick_{cluster}").select(pick))
    if free:
        run(at.text_input(key=f"rv_free_{cluster}").input(free))
    if basis:
        run(at.button_group(key=f"rv_basis_{cluster}").set_value(basis))
    return run(at.button(key=f"rv_save_{cluster}").click())


def accept(at, cluster: str):
    at = go(at, "markers", cluster)
    run(at.toggle(key=f"rv_reveal_{cluster}").set_value(True))
    return run(at.button(key=f"rv_accept_{cluster}").click())


def defer(at, cluster: str):
    at = go(at, "markers", cluster)
    return run(at.button(key=f"rv_defer_{cluster}").click())


# --------------------------------------------------------------------------- #
# A corpus built in the biologist's own LLM (IMPL §6.1): synthetic papers, resolved by
# a stubbed Europe PMC lookup (``isolated.ids``), never the network.
# --------------------------------------------------------------------------- #
def literature(isolated, monkeypatch, study, tmp: Path, claims: list[tuple[str, str, str, str]],
               extra_rows: list[dict] = (), name: str = "corpus.jsonl") -> Path:
    """``claims``: (paper id, gene, cell type, 'agree' | 'dissent'). Each paper gets a
    verbatim abstract and tissue setting; returns the JSONL a client would write."""
    import json

    from panoscope.corpus import build as B
    from panoscope.corpus import exchange as X
    from panoscope.corpus.store import SourceRecord

    monkeypatch.setattr(X.vocab, "hgnc", lambda g: None)    # symbol only, offline
    rows: list[dict] = [{"kind": "header", "format_version": 1, "study_id": study.id,
                         "brief_hash": B.brief_hash(study), "client": "Claude Desktop",
                         "provider": "Anthropic", "model": "some-model",
                         "run_date": "2026-11-07"}]
    for i, (pid, gene, ct, stance) in enumerate(claims):
        who = ct.replace("_", " ").lower()
        span = (f"{gene} marks {who}" if stance == "agree" else f"{who} lacked {gene} expression")
        isolated.ids[pid] = SourceRecord(
            uri=pid, title=f"paper {i}", abstract=f"In human breast cancer, {span}. "
            "breast tumour samples were profiled.", via="Europe PMC", year=2024 + i)
        rows += [{"kind": "paper", "id": pid, "channel": "C1", "disposition": "READ:1:0"},
                 {"kind": "statement", "paper": pid, "gene": gene, "cell_type": ct,
                  "predicate": "MARKS" if stance == "agree" else "NOT_EXPRESSED_IN",
                  "span": span, "section": "abstract"},
                 {"kind": "setting", "paper": pid, "field": "tissue",
                  "span": "breast tumour samples"}]
    keys = dict.fromkeys((g, ct) for _, g, ct, _ in claims)
    rows += [{"kind": "query", "query": f'"{g}" AND breast', "source": "Europe PMC",
              "channel": "C1", "gene": g, "cell_type": ct, "status": "OK",
              "total": sum(k == (g, ct) for k in ((x[1], x[2]) for x in claims)),
              "retrieved": sum(k == (g, ct) for k in ((x[1], x[2]) for x in claims))}
             for g, ct in keys]
    rows += list(extra_rows)
    p = tmp / name
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def import_corpus(at, study_id: str, path: Path):
    """The corpus step after Build: import the client's file; returns the AppTest."""
    at.session_state["ob_corpus"] = study_id
    at = go(at, "upload")
    run(at.text_input(key="ob_corpus_file").input(str(path)))
    return run(at.button(key="ob_corpus_import").click())
