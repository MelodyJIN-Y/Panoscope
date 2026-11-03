"""Which technology's reader a study gets, and why it is never guessed.

Both composition roots -- ``pipeline/records.py`` and ``pipeline/stages/enrichment``
-- constructed ``XeniumAdapter()`` as a literal, each with a comment saying no
registry existed. That made ARCHITECTURE §4's "the core does not change when a
member is added" false for platforms specifically, and the failure it set up is the
quiet kind: the adapter decides the capability frames, so a MERSCOPE store read
through the Xenium adapter would not raise. It would answer, about a panel that was
never run, and every measurement state in the study would be downstream of that.

So the tests here are mostly about REFUSAL. There is one registered platform today,
which means the interesting assertions are all negative -- an unknown platform, an
undeclared platform, a mis-registered path -- because with a single entry a positive
test passes whether the lookup works or the answer is hardcoded.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from panoscope import platforms
from panoscope.data import layout
from panoscope.data.study import Study
from panoscope.types.uispec import DuplicateRegistration

ROOTS = (
    Path("panoscope/pipeline/records.py"),
    Path("panoscope/pipeline/stages/enrichment.py"),
)


@pytest.mark.parametrize("study_id", sorted(layout.available_studies()))
def test_every_committed_study_resolves_an_adapter_that_claims_it(
        study_id: str) -> None:
    """The resolved adapter declares the name the study asked for.

    Not merely "an adapter came back": the name check is what makes this a lookup
    rather than a default. A single-entry dict returns the same object for every
    key if the lookup is wrong, and nothing else here would notice.
    """
    study = Study.open(study_id)
    adapter = platforms.for_study(study)
    assert adapter.name == str(study.platform).strip().lower()


def test_an_unregistered_platform_refuses_instead_of_defaulting() -> None:
    """The headline refusal. There is no safe fallback, so there is no fallback.

    A study whose platform nobody wrote an adapter for must stop. Returning "the
    only registered adapter" would read every frame off the wrong technology, and
    the record would look exactly like a successful build.
    """
    with pytest.raises(platforms.UnregisteredPlatform) as e:
        platforms.get("merscope")
    assert "merscope" in str(e.value)
    assert "xenium" in str(e.value), "the refusal should name what IS registered"


def test_a_study_declaring_no_platform_refuses(tmp_path: Path) -> None:
    """§6.3 blocks the study on platform. Inferring it is the forbidden guess.

    A manifest with no ``platform`` is a study nobody declared the technology for.
    Guessing from a filename, or from which files happen to be present, is exactly
    what the blocking rule exists to prevent.
    """
    d = tmp_path / "studies" / "no_platform"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(
        json.dumps({"dataset_id": "no_platform", "tissue": "human kidney"}),
        encoding="utf-8")

    with pytest.raises(platforms.UnregisteredPlatform) as e:
        platforms.for_study(Study(id="no_platform", data_root=tmp_path))
    assert "declares no platform" in str(e.value)


def test_a_misregistered_path_is_caught_at_first_use(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """A typo in the registry must not silently hand over another reader.

    ``get`` checks that the resolved object declares the name it was registered
    under. Without that, registering the Xenium adapter under ``"merscope"`` -- a
    copy-paste away -- would succeed, and every study on that platform would be
    read by the wrong one with no error anywhere.
    """
    monkeypatch.setitem(platforms.ADAPTERS, "merscope",
                        "panoscope.platforms.xenium:XeniumAdapter")
    with pytest.raises(platforms.UnregisteredPlatform) as e:
        platforms.get("merscope")
    assert "declares name 'xenium'" in str(e.value)


def test_registering_a_duplicate_raises() -> None:
    """Last-write-wins between two registrations is decided by import order, which
    is decided by nothing -- and the loser fails far from the cause."""
    with pytest.raises(DuplicateRegistration):
        platforms.register("Xenium", "panoscope.platforms.xenium:XeniumAdapter")


def test_importing_the_registry_imports_no_adapter() -> None:
    """The constraint the module docstring states, asserted rather than trusted.

    Every value in ``ADAPTERS`` is a ``"module:attr"`` string precisely so that
    importing the package costs nothing and drags in no platform's read
    dependencies. The day someone "simplifies" it to live objects, this fails.
    """
    for key, value in platforms.ADAPTERS.items():
        assert isinstance(value, str), (
            f"{key} is registered as a live object; importing the registry now "
            "imports every platform's dependencies"
        )
        assert ":" in value, f"{key} -> {value!r} is not a 'module:attr' path"


@pytest.mark.parametrize("root", ROOTS, ids=lambda p: p.name)
def test_no_composition_root_names_a_concrete_adapter(root: Path) -> None:
    """The regression this registry exists to prevent, guarded as source text.

    A behavioural test cannot catch it: a root that constructs ``XeniumAdapter()``
    directly passes every test in the suite while there is one platform. A source
    guard fails the moment the literal reappears, which is when it costs nothing
    to undo.
    """
    source = root.read_text(encoding="utf-8")
    offending = [
        line.strip() for line in source.splitlines()
        if "XeniumAdapter(" in line and not line.lstrip().startswith("#")
    ]
    assert not offending, (
        f"{root} constructs an adapter directly: {offending}. The study's manifest "
        "says which technology produced it; use platforms.for_study(study)."
    )
