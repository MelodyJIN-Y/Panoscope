"""Reference data: it resolves, it does not retrieve, and it never supplies a value.

``graph.py`` holds ``Node``, ``Edge`` and the three guards that make "reference data
may never supply a value, count as agreement, or corroborate a measurement"
structural rather than documented. ``sources.py`` holds the registry, the licence
gate, and PanglaoDB.

Importing this package touches no file and no network. Sources are registered by the
study build, not here -- a prepopulated registry would have to name a tissue, and the
demo is breast while the second study is kidney.
"""
