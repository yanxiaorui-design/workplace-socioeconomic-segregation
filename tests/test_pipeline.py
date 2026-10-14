"""Exercise the public CLI from clean inputs and verify its numerical products."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_complete_demo(tmp_path):
    subprocess.run([sys.executable, str(ROOT/'main.py'), 'demo', '--output', str(tmp_path)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT/'scripts/verify_demo.py'), '--output', str(tmp_path)], cwd=ROOT, check=True)
