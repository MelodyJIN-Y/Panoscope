"""The guard that watches data/studies/ -- tested on a tree of its own.

``data/studies/`` is a committed INPUT. Twice in three rounds a test wrote into it and
``git status`` stayed silent, because the runtime paths under ``interp/`` are gitignored
by design. ``tree_guard.snapshot_tree`` / ``tree_guard.diff_tree`` are what notice; this
file is what notices when THEY stop noticing.

The cases are deliberately split by which design of guard catches them:

* an ADDED file -- both leaks so far were this shape, and a cheap path-set comparison
  catches it. This is the easy half.
* a file MODIFIED IN PLACE, same path and same byte length -- the shape of the second
  leak, a probe string written over ``xenium_hbreast_sample1/interp/summary_edits.json``,
  a path that can legitimately already be on disk. Only the content digest sees it, and
  a guard rewritten to compare path sets alone reports it clean. This is the half that
  has to be pinned, because it is the half a future simplification would quietly drop.

Everything here runs against ``tmp_path``. Nothing in this file touches the real tree --
which would be a fine way to make the guard it is testing fire.
"""

from __future__ import annotations

from pathlib import Path


def _tree(root: Path) -> Path:
    """A miniature study tree: a tracked input and a gitignored runtime sibling."""
    interp = root / "demo" / "interp"
    interp.mkdir(parents=True)
    (root / "demo" / "manifest.json").write_text('{"dataset_id": "demo"}')
    (interp / "summary_edits.json").write_text('{"c1": "aaaa"}')
    return root


def test_an_untouched_tree_reports_nothing(tree_guard, tmp_path):
    before = tree_guard.snapshot_tree(_tree(tmp_path))
    assert tree_guard.diff_tree(before, tree_guard.snapshot_tree(tmp_path)) == []


def test_an_absent_root_is_a_no_op_rather_than_an_error(tree_guard, tmp_path):
    """What the out-of-tree CI job sees: tests copied out, checkout moved aside."""
    gone = tmp_path / "nothing-here"
    assert tree_guard.snapshot_tree(gone) == {}
    assert tree_guard.diff_tree(tree_guard.snapshot_tree(gone), tree_guard.snapshot_tree(gone)) == []


def test_a_new_file_is_named(tree_guard, tmp_path):
    """The easy half: the shape of both leaks so far, caught by any design."""
    before = tree_guard.snapshot_tree(_tree(tmp_path))
    (tmp_path / "half_prepped" / "interp").mkdir(parents=True)
    (tmp_path / "half_prepped" / "interp" / "chat_threads.json").write_text("{}")
    assert tree_guard.diff_tree(before, tree_guard.snapshot_tree(tmp_path)) == [
        "+ half_prepped/interp/chat_threads.json"
    ]


def test_a_deleted_file_is_named(tree_guard, tmp_path):
    before = tree_guard.snapshot_tree(_tree(tmp_path))
    (tmp_path / "demo" / "interp" / "summary_edits.json").unlink()
    # Two lines, not one: removing the only file leaves ``demo/interp/`` empty, and
    # an empty directory is its own entry. See the next test for why it has to be.
    assert tree_guard.diff_tree(before, tree_guard.snapshot_tree(tmp_path)) == [
        "+ demo/interp/",
        "- demo/interp/summary_edits.json",
    ]


def test_a_directory_left_behind_with_no_file_in_it_is_named(tree_guard, tmp_path):
    """The shape a FAIL-SOFT writer leaves, and the one a file-only snapshot misses.

    ``ui/chat_store.save_all`` mkdir -p's ``<study>/interp/`` and then swallows any
    write error, deliberately -- losing a chat transcript must never break the app.
    When that write fails it leaves a directory under a study id that exists nowhere
    and not one file to notice it by, which the next ``panoscope build`` still reads
    as a study. A snapshot that records only ``p.is_file()`` reports that tree clean.
    """
    before = tree_guard.snapshot_tree(_tree(tmp_path))
    (tmp_path / "half_prepped" / "interp").mkdir(parents=True)
    assert tree_guard.diff_tree(before, tree_guard.snapshot_tree(tmp_path)) == [
        "+ half_prepped/interp/"
    ]


def test_residue_already_on_disk_is_reported_even_when_nothing_changed(tree_guard):
    """The guard's blind spot, pinned: a leak present at BOTH ends reads as clean.

    ``diff_tree`` compares the tree against itself, so it sees a leak only while it
    is being made. Once the file is on disk every later run starts and ends with it
    there and the comparison is empty -- which is how four fixes in one round came
    to be verified against a green suite. ``untracked_paths`` asks the other
    question, and it is the one that has to be asked of the tree as FOUND.
    """
    assert tree_guard.diff_tree({"a": "1"}, {"a": "1"}) == [], (
        "sanity: an unchanged tree is, by the change half, clean"
    )
    # The real tree, as this run found it. Empty is the only acceptable answer, and
    # an empty list here is what the change half can never tell you.
    assert tree_guard.untracked_paths() == [], (
        "data/studies/ carries untracked residue; the change-based guard is blind "
        "to it for the whole of this session"
    )


def test_an_in_place_overwrite_of_the_same_length_is_named(tree_guard, tmp_path):
    """The hard half, and the whole reason the snapshot hashes content.

    Same path, same byte count -- only the bytes differ, exactly as when a verifier
    wrote a probe string over a real ``summary_edits.json``. A path-set guard is green
    here, and so is a size-and-mtime guard whose mtime got restored. Note what that
    one leak then cost: while the probe file sat there, ``tests/test_summary.py``'s
    own guard skipped itself, so the one test watching for the violation was disabled
    by the violation.
    """
    tree = _tree(tmp_path)
    target = tree / "demo" / "interp" / "summary_edits.json"
    original = target.read_bytes()
    before = tree_guard.snapshot_tree(tree)

    target.write_bytes(b'{"c1": "PROBE"}'[: len(original)].ljust(len(original), b" "))
    assert target.stat().st_size == len(original), "the point is an equal-length rewrite"

    assert tree_guard.diff_tree(before, tree_guard.snapshot_tree(tree)) == [
        "~ demo/interp/summary_edits.json"
    ]
    # And a path-set comparison -- the cheaper design -- sees nothing to report.
    after = tree_guard.snapshot_tree(tree)
    assert set(after) == set(before)


def test_the_guard_armed_itself_for_this_run(tree_guard):
    """Wiring, not logic: the helpers above can be perfect and still never be called.

    ``pytest_configure`` populates the baseline before collection. If a rename or a
    refactor unhooks it, every assertion in this file still passes while the real tree
    goes unwatched -- which is the failure mode this guard exists to end.
    """
    assert tree_guard._TREE_BASELINE is not None, (
        "tree_guard.pytest_configure did not run; the data-tree guard is not armed"
    )
    assert "manifest.json" in " ".join(tree_guard._TREE_BASELINE), (
        "the baseline is empty -- data/studies/ was not found from "
        f"{tree_guard.DATASETS_ROOT}"
    )


def test_the_failure_message_says_what_to_do(tree_guard):
    """A guard whose message does not name the fix gets `-p no:cacheprovider`'d.

    Pinned because the message IS the feature: nine times out of ten the cause is a
    missing ``layout.clear_cache()`` after a ``PANOSCOPE_DATA_ROOT`` setenv, and a bare
    "the tree changed" sends the reader looking at `git status`, which is silent.
    """
    msg = tree_guard._TREE_DIRTY.format(paths="    + a/b.json")
    assert "a/b.json" in msg
    assert "layout.clear_cache()" in msg
    assert "PANOSCOPE_DATA_ROOT" in msg
    assert "git status" in msg
