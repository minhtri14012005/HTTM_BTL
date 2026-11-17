import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('facewatch_launcher_test', ROOT / 'run.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


@pytest.mark.skipif(os.name != 'nt', reason='Windows browser launch')
def test_diagnostic_opens_its_own_chrome_window_with_exact_local_path():
    path = '/?diagnostic=camo&close_after=1'
    with patch.object(launcher.Path, 'is_file', return_value=True), patch.object(launcher.subprocess, 'Popen') as launch:
        assert launcher.open_interface(path)
    args = launch.call_args.args[0]
    assert args[0].lower().endswith('chrome.exe')
    assert args[1:] == ['--new-window', '--start-maximized', launcher.URL + path]
    assert launch.call_count == 1


@pytest.mark.parametrize('path', ['https://example.com', '//example.com', '/\\example.com'])
def test_browser_launch_rejects_nonlocal_paths(path):
    with patch.object(launcher.subprocess, 'Popen') as launch:
        with pytest.raises(ValueError):
            launcher.open_interface(path)
    launch.assert_not_called()


def test_launcher_cli_has_a_single_open_path_option():
    result = subprocess.run([sys.executable, str(ROOT / 'run.py'), '--help'], capture_output=True)
    assert result.returncode == 0
    assert b'--open-path' in result.stdout
