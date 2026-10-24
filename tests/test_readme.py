"""The quickstart in the README and on the docs front page runs and says true."""

import re
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def quickstart(path: Path) -> str:
    text = path.read_text()
    section = text.split("## Quickstart", 1)[1]
    block = re.search(r"```python\n(.*?)```", section, re.S)
    assert block, f"no python block under Quickstart in {path.name}"
    return block.group(1)


@pytest.mark.parametrize("page", ["README.md", "docs/index.md"])
def test_quickstart_shows_the_coupling_not_an_artefact(page):
    path = ROOT / page
    if not path.exists():  # the sdist ships the README but not the docs
        pytest.skip(f"{page} not shipped")
    namespace: dict[str, Any] = {}
    exec(quickstart(path), namespace)
    # what the comment promises; unequal noise once made this 0.80 (#104)
    assert 0.6 < namespace["dPTE"][0, 1] < 0.67
