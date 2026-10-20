"""
Integration tests for Flask Web Application routes and endpoints.
"""

import unittest
import app


class TestFlaskWebApp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.app.config['TESTING'] = True
        cls.client = app.app.test_client()

    def test_index_route(self):
        """Verify the main UI index loads with HTTP 200."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'FRUIT', response.data)

    def test_api_samples(self):
        """Verify /api/samples returns 10 catalog fruits."""
        response = self.client.get('/api/samples')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('samples', data)
        self.assertEqual(len(data['samples']), 10)

    def test_api_metrics(self):
        """Verify /api/metrics returns valid evaluation data."""
        response = self.client.get('/api/metrics')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('test_accuracy', data)
        self.assertIn('macro_f1', data)
        self.assertEqual(data['classes_count'], 10)

    def test_api_predict_empty_payload(self):
        """Verify /api/predict rejects empty requests with HTTP 400."""
        response = self.client.post('/api/predict')
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn('error', data)


if __name__ == '__main__':
    unittest.main()
