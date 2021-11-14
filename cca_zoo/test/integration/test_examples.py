import subprocess

import pytest


def execute_commands(cmds):
    for cmd in cmds:
        try:
            out = subprocess.check_output(cmd, shell=True).decode("utf-8")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Command {cmd} return with err {e.returncode} {e.output}"
            )


class TestTutorial:
    @pytest.mark.parametrize(
        "name",
        [
            "plot_sparse_cca",
        ],
    )
    def test_registry(self, name: str):
        cmds = [f"python examples/{name}.py"]
        execute_commands(cmds)
