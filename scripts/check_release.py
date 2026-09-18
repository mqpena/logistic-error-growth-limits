#!/usr/bin/env python3
"""Run all existing compact regression tests, including function-style tests.

Set PRL_SQG_DATA to the extracted bundle's data/raw directory to enable the
existing large-artifact provenance test. No simulations or resampling run.
"""
from pathlib import Path
import inspect
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.validity_limit_v1.tests import test_m1_v1, test_mtb_v1


def main():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for module in (test_m1_v1, test_mtb_v1):
        suite.addTests(loader.loadTestsFromModule(module))
        for name, test in inspect.getmembers(module, inspect.isfunction):
            if name.startswith('test_') and test.__module__ == module.__name__:
                suite.addTest(unittest.FunctionTestCase(test))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
