"""J9: two sections clustered jointly; the clustering mode must be declared (S9)."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tests.journeys.flow import build_and_start, declare, go, paste, upload
from tests.ui.screen import screen_text

SYNTH = Path(__file__).resolve().parents[2] / "data" / "studies" / "synthetic_multisample_fixture"


def _blocking(at) -> str:
    return at.code[0].value.split("BLOCKING", 1)[1].split("Optional", 1)[0]


@pytest.mark.req("ARCH-6.3-02", "PLAN-P1.2-01", "ARCH-9.10-01", "UI-4.5-01")
def test_two_samples_block_until_the_mode_is_declared_and_divergence_is_flagged(
        isolated, tmp_path):
    """Given a folder with samples/sampleA and samples/sampleB (synthetic fixture),
    When the biologist pastes it, Then clustering_mode blocks with tissue, naming both
    samples; When they type 'maybe', Then it is refused and the study stays blocked;
    When they declare joint and build, Then c1 shows the replicate comparison across
    both samples, flags a marker that read differently, and says agreement would not
    have raised confidence."""
    folder = tmp_path / "two_sections"
    shutil.copytree(SYNTH / "inputs", folder)
    samples = sorted(p.name for p in (folder / "samples").iterdir())

    at = paste(upload(), folder)
    block = _blocking(at)
    assert "clustering_mode" in block and "tissue" in block
    assert all(s in at.code[0].value for s in samples)
    assert [w.value for w in at.warning] == ["Blocked until declared: tissue, clustering_mode"]

    at = declare(at, tissue="synthetic fixture", clustering_mode="maybe")
    assert "'maybe' is not one of joint, independent, integrated" in " ".join(
        e.value for e in at.error) + at.code[0].value
    assert "ob_build" not in {x.key for x in at.button}

    at = build_and_start(declare(at, clustering_mode="joint"))
    text = screen_text(go(at, "markers", "c1"))
    assert "replicate concordance compared" in text and all(s in text for s in samples)
    assert "read differently across the samples" in text
    assert "agreement would not have raised confidence" in text
