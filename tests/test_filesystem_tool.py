from pathlib import Path

import pytest

from k2do.agent.tools.filesystem import WriteFileTool


@pytest.mark.asyncio
async def test_write_file_relative_path_inside_workspace(tmp_path: Path) -> None:
    tool = WriteFileTool(allowed_dir=tmp_path)
    result = await tool.execute(path="demo.txt", content="ok")

    assert "Successfully wrote" in result
    assert (tmp_path / "demo.txt").read_text(encoding="utf-8") == "ok"


@pytest.mark.asyncio
async def test_write_file_rejects_path_traversal(tmp_path: Path) -> None:
    tool = WriteFileTool(allowed_dir=tmp_path)
    result = await tool.execute(path="../escape.txt", content="bad")

    assert "outside allowed directory" in result
