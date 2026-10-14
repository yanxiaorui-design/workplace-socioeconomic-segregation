"""Check numerical outputs against the bundled synthetic reference results."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def compare_json(actual, expected):
    """Compare nested numeric summaries with a tolerance for linear algebra versions."""
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            compare_json(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            compare_json(a, e)
    elif isinstance(expected, (float, int)):
        assert np.isclose(actual, expected, rtol=1e-7, atol=1e-10)
    else:
        assert actual == expected


def verify(output):
    """Fail on missing or changed reference tables; ignore runtime metadata."""
    expected = json.loads((ROOT / 'data/expected_results.json').read_text(encoding="utf-8"))
    checked = 0
    for relative_path, reference in expected.items():
        produced = output / relative_path
        if reference['format'] == 'table':
            table = pd.DataFrame(reference['data'], columns=reference['columns'])
            pd.testing.assert_frame_equal(pd.read_csv(produced), table, check_exact=False, rtol=1e-7, atol=1e-10)
        else:
            compare_json(json.loads(produced.read_text(encoding="utf-8")), reference['value'])
        checked += 1
    if checked == 0:
        raise AssertionError('No expected outputs were found')
    print(f'Verified {checked} synthetic reference outputs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/demo')
    verify(parser.parse_args().output)
