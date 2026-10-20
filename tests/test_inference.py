"""
Unit tests for FruitInferencePipeline and deep feature inference.
"""

import os
import unittest
from PIL import Image
from src.inference_engine import FruitInferencePipeline


class TestInferencePipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model_path = "models/svm_model.pkl"
        if os.path.exists(cls.model_path):
            cls.pipeline = FruitInferencePipeline(model_path=cls.model_path)
        else:
            cls.pipeline = None

    def test_pipeline_initialization(self):
        """Verify pipeline loads correct model metadata and 10 fruit categories."""
        if self.pipeline is None:
            self.skipTest("models/svm_model.pkl not present for inference test.")

        self.assertEqual(len(self.pipeline.class_names), 10)
        self.assertEqual(self.pipeline.backbone_name, 'resnet50')
        self.assertEqual(self.pipeline.W_all.shape, (45, 2048))
        self.assertEqual(self.pipeline.intercepts_all.shape, (45,))

    def test_prediction_output_schema(self):
        """Verify inference returns all required diagnostic fields on dummy image."""
        if self.pipeline is None:
            self.skipTest("models/svm_model.pkl not present for inference test.")

        dummy_img = Image.new('RGB', (224, 224), color=(200, 50, 50))
        results = self.pipeline.predict(dummy_img, top_k=3)

        self.assertIn('predicted_class', results)
        self.assertIn('confidence', results)
        self.assertIn('top_classes', results)
        self.assertIn('all_classes', results)
        self.assertIn('pairwise_votes', results)
        self.assertIn('feature_stats', results)
        self.assertIn('timing', results)

        self.assertEqual(len(results['top_classes']), 3)
        self.assertIn(results['predicted_class'], self.pipeline.class_names)


if __name__ == '__main__':
    unittest.main()
