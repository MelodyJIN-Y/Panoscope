"""Deterministic interpretation. Zero LLM, zero IO, zero clock.

Every public function here returns ``Traced[T]`` and accepts no ``Trace``:
decisions flow OUT, never in, so there is no seam a caller could reach through to
inject a narrator into the analysis layer. ``tests/test_contracts.py`` enforces
both halves by signature and by source scan.
"""
