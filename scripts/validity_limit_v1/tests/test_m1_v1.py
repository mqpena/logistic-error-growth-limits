from __future__ import annotations

import tempfile
import unittest
import json
import os
from pathlib import Path

import numpy as np

from scripts.validity_limit_v1 import conventions_v1 as cv
from scripts.validity_limit_v1.frozen_regression_v1 import recompute
from scripts.validity_limit_v1.io_v1 import load_npz_verified, parse_metadata, verify_hash

ROOT = Path(__file__).resolve().parents[3]
NULL_PATH = ROOT / "results/v8_pilot_null_gate_v2_20260720.json"


class ConventionTests(unittest.TestCase):
    def test_metric_identity(self):
        rng = np.random.default_rng(20260809)
        u = rng.normal(size=32) + 1j * rng.normal(size=32)
        f = rng.normal(size=32) + 1j * rng.normal(size=32)
        et, ef, c = cv.Et(u), cv.Ef(f), cv.C(u, f)
        error = cv.epsilon(u, f)
        self.assertAlmostEqual(error, et + ef - 2 * c, places=12)
        self.assertAlmostEqual(cv.x_C_from_moments(et, ef, c),
                               cv.x_C_from_epsilon(et, ef, error), places=12)
        self.assertAlmostEqual(cv.full_norm_error_from_half_energy(error), 2 * error)

    def test_equal_energy_complementarity(self):
        rng = np.random.default_rng(9)
        u = rng.normal(size=24)
        f = rng.normal(size=24)
        f *= np.linalg.norm(u) / np.linalg.norm(f)
        et, ef, c = cv.Et(u), cv.Ef(f), cv.C(u, f)
        error = cv.epsilon(u, f)
        self.assertAlmostEqual(et, ef, places=12)
        self.assertAlmostEqual(cv.x_C_from_moments(et, ef, c),
                               cv.rho_from_moments(et, ef, c), places=12)
        self.assertAlmostEqual(cv.x_C_from_moments(et, ef, c),
                               1 - error / (et + ef), places=12)

    def test_zero_normalizer_fails(self):
        with self.assertRaises(ValueError):
            cv.normalized_production(np.array([1.]), np.array([0.]), np.array([1.]))

    def test_deliberately_doubled_epsilon_fails(self):
        et, ef, c = np.array([2.0]), np.array([2.0]), np.array([1.0])
        correct = et + ef - 2 * c
        with self.assertRaises(RuntimeError):
            cv.validate_metric_identity(et, ef, c, 2 * correct)


class ProvenanceTests(unittest.TestCase):
    def test_altered_artifact_fails(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "x"
            path.write_bytes(b"altered")
            with self.assertRaises(RuntimeError):
                verify_hash(path, "0" * 64)

    def test_frozen_null_regression(self):
        result = recompute(NULL_PATH)
        self.assertEqual(result["counts_ge_observed"],
                         {"time_block": 109, "cross_truth": 113, "phase_scramble": 0})
        self.assertAlmostEqual(result["p_time_block"], 0.011, places=15)
        self.assertAlmostEqual(result["p_cross_truth"], 113 / 384, places=15)
        self.assertAlmostEqual(result["p_phase_scramble"], 0.001, places=15)

    def test_all_cluster_hashes_shapes_and_metadata(self):
        config_path = ROOT / "scripts/validity_limit_v1/config_v1.json"
        config = json.loads(config_path.read_text())
        configured = os.environ.get("PRL_SQG_DATA")
        if not configured:
            self.skipTest("set PRL_SQG_DATA to run large-artifact provenance checks")
        cluster_root = Path(configured).expanduser().resolve()
        for relative, expected in config["clusters"]:
            with self.subTest(relative=relative):
                with load_npz_verified(cluster_root / relative, expected) as npz:
                    self.assertEqual(npz["scalar_time"].shape, (733,))
                    self.assertEqual(npz["locality_time"].shape, (125,))
                    self.assertEqual(npz["field_time"].shape, (125,))
                    self.assertEqual(npz["Et"].shape, (733, 2, 6))
                    self.assertEqual(npz["C"].shape, (733, 2, 6))
                    metadata = parse_metadata(npz)
                    self.assertIsInstance(metadata, dict)


if __name__ == "__main__":
    unittest.main()
