"""Offline checks for API row alignment and input limits; no credentials/network."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from embed_foundationalassist_openai import batches, response_vectors
from cluster_relabel_foundationalassist import fit_labels, induced_arrays


class APIContract(unittest.TestCase):
    def response(self, data, model='test'):
        return SimpleNamespace(model=model, data=[SimpleNamespace(index=i, embedding=v) for i, v in data])

    def test_response_indices_restore_request_order(self):
        response = self.response([(1, [0, 1]), (0, [1, 0])])
        np.testing.assert_array_equal(response_vectors(response, 2, 2, 'test'), np.eye(2))

    def test_invalid_response_is_rejected(self):
        cases = [[(0, [1, 0]), (0, [0, 1])], [(1, [0, 1])],
                 [(0, [float('nan'), 0]), (1, [0, 1])],
                 [(0, [1, 0, 0]), (1, [0, 1, 0])], [(0, [0, 0]), (1, [0, 1])]]
        for data in cases:
            with self.subTest(data=data), self.assertRaises(ValueError):
                response_vectors(self.response(data), 2, 2, 'test')
        with self.assertRaises(ValueError):
            response_vectors(self.response([(0, [1, 0])], 'wrong'), 1, 2, 'test')

    def test_raw_API_values_are_preserved_without_renormalizing(self):
        vectors = np.array([[1.0007, 0], [0, .9994]], dtype=np.float32)
        response = self.response([(0, vectors[0]), (1, vectors[1])])
        np.testing.assert_array_equal(response_vectors(response, 2, 2, 'test'), vectors)

    def test_batch_limits_preserve_all_inputs(self):
        rows = [{'problem_text': 'é' * 40, 'id': i} for i in range(300)]
        output = list(batches(rows, max_items=128, max_bytes=800))
        self.assertEqual([r for b in output for r in b], rows)
        self.assertTrue(all(len(b) <= 128 and sum(len(r['problem_text'].encode()) for r in b) <= 800 for b in output))

    def test_empty_and_long_inputs_fail_without_truncation(self):
        for text in ['', '  ', 'a' * 8192]:
            with self.subTest(length=len(text)), self.assertRaises(ValueError):
                list(batches([{'problem_text': text}]))


class CPUContract(unittest.TestCase):
    def test_relabeling_preserves_padding_and_all_shared_fields(self):
        shared = {'questions': np.array([[0, 1, -1], [2, -1, -1]]),
                  'original_concepts': np.array([[1, 0, -1], [1, -1, -1]]),
                  'responses': np.array([[1, 0, -1], [0, -1, -1]]),
                  'event_id': np.array([[10, 11, -1], [12, -1, -1]]),
                  'target_mask': np.array([[False, True, False], [False, False, False]])}
        result = induced_arrays(shared, np.array([2, 1, 0]))
        np.testing.assert_array_equal(result['induced_concepts'], [[2, 1, -1], [0, -1, -1]])
        for key in shared:
            np.testing.assert_array_equal(result[key], shared[key])

    def test_cpu_cluster_refit_is_deterministic(self):
        values = np.random.default_rng(12).normal(size=(40, 8)).astype(np.float32)
        config = {'pca_components': 4, 'pca_svd_solver': 'full', 'pca_whiten': False,
                  'random_state': 42, 'n_clusters': 4, 'n_init': 10, 'algorithm': 'lloyd',
                  'max_iter': 300, 'tol': .0001}
        a, scores_a, _, _ = fit_labels(values, config)
        b, scores_b, _, _ = fit_labels(values.copy(), config)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(scores_a, scores_b)
        self.assertEqual(len(set(a)), 4)


if __name__ == '__main__':
    unittest.main()
