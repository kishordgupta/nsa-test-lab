import os
from pathlib import Path


def test_gui_launchers_are_executable():
    repo = Path(__file__).resolve().parents[1]
    assert os.access(repo / "bin" / "nsa-test-lab-gui", os.X_OK)
    assert os.access(repo / "NSA Test Lab.command", os.X_OK)
    assert os.access(repo / "NSA Test Lab.app" / "Contents" / "MacOS" / "NSA Test Lab", os.X_OK)
