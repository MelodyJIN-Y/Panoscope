"""Start the app with the scripted model client in place of Anthropic's (plan 7.4).

``python -m tests.browser.launch <streamlit run options>``. The replies come from
``PANOSCOPE_FAKE_REPLIES`` (a JSON list of answer strings) and the ids that resolve as
real from ``PANOSCOPE_FAKE_IDS``. Streamlit runs the app in this process, so the
patched ``panoscope.agent.loop`` is the one it uses; the app gains no test hook.
"""
import json
import os
import sys

import pytest

from tests.browser.conftest import APP
from tests.fixtures.fake_model import install

mp = pytest.MonkeyPatch()
client = install(mp)
client.script(*json.loads(os.environ.get("PANOSCOPE_FAKE_REPLIES", "[]")))
client.real_ids |= set(json.loads(os.environ.get("PANOSCOPE_FAKE_IDS", "[]")))

from panoscope.agent import tools  # after the client is in place

mp.setattr(tools, "_LITERATURE_SEARCH_FN", lambda q: [])   # no PubMed from a test

from streamlit.web import cli

sys.argv = ["streamlit", "run", str(APP), *sys.argv[1:]]
sys.exit(cli.main())
