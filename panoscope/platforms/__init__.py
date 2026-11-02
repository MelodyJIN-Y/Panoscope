"""Platform adapters: how one technology's store answers the questions
interpretation depends on.

Deliberately empty of re-exports. ``panoscope.platforms.xenium`` is imported by name
by whatever registers it, because a package ``__init__`` that imports every adapter
makes adding a platform a two-file change and makes importing *one* adapter drag in
the dependencies of all of them -- the opposite of what a registry is for
(``types/contracts.py``: "adding a platform cannot break a method").

Nothing here runs at import. ``tests/test_import_surface.py`` requires that importing
the package read no dataset file, and an adapter that resolved a study at module
scope would freeze every path to whichever study happened to be active first -- the
bug ``data/study.py`` was written to kill.
"""
