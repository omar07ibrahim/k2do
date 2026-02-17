from pathlib import Path

import pytest

from k2do.agent.tools.shell import ExecTool


@pytest.mark.asyncio
async def test_exec_blocks_working_dir_outside_workspace(tmp_path: Path) -> None:
    tool = ExecTool(
        working_dir=str(tmp_path),
        restrict_to_workspace=True,
        workspace_root=str(tmp_path),
    )

    result = await tool.execute("pwd", working_dir=str(tmp_path.parent))
    assert "working_dir outside workspace" in result
