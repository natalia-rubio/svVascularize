"""
This file launches the testing suite for the svVascularize (svv)
package. You can run the test suite with the following command:

`python -m svv.tests [pytest args]

The real test files are in the top-level ``tests/`` folder of the
source code and not packaged within the distibution wheels. If you
would like to run the tests, you will require the full repo.
"""
from __future__ import annotations
import pathlib
import sys
import pytest

def main() -> None:
    # Locate the source package root wherever it is installed.
    root = pathlib.Path(__file__).resolve().parents[2]
    tests_dir = root / "test"

    if not tests_dir.is_dir():
        # This installation does not contain the tests suite
        # (probably because this is a wheel installation)
        print(
            "    The full test-suite isn’t packaged in the installed wheel.\n"
            "    Clone the repository if you want to run it:\n"
            "       git clone https://github.com/SimVascular/svVascularize.git && cd svv && "
            "python -m svv.tests"
        )
        sys.exit(1)

    sys.exit(pytest.main([str(tests_dir), *sys.argv[1:]]))

if __name__ == "__main__":
    main()