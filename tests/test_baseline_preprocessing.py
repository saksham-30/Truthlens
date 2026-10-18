import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.train_baseline import predict_baseline


class BaselinePreprocessingTests(unittest.TestCase):
    def test_inference_normalizes_text_before_vectorization(self):
        vectorizer = Mock()
        vectorizer.transform.return_value = 'features'
        model = Mock()
        model.predict.return_value = np.array([1])
        model.predict_proba.return_value = np.array([[0.2, 0.8]])

        with patch('src.train_baseline._get_top_features', return_value={}):
            result = predict_baseline('  Sample\n text  ', vectorizer, model)

        vectorizer.transform.assert_called_once_with(['Sample text'])
        self.assertEqual(result['prediction'], 1)
        self.assertEqual(result['ai_prob'], 0.8)

    def test_empty_normalized_text_is_rejected(self):
        vectorizer = Mock()
        model = Mock()

        with self.assertRaisesRegex(ValueError, 'empty after preprocessing'):
            predict_baseline(' \n ', vectorizer, model)

        vectorizer.transform.assert_not_called()


if __name__ == '__main__':
    unittest.main()