"""
Unit tests for data loader and preprocessing utilities.
"""

import unittest
from torchvision import transforms
from src.data_loader import (
    CANONICAL_CLASSES,
    CANONICAL_CLASS_MAP,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    get_default_transforms
)


class TestDataLoader(unittest.TestCase):

    def test_canonical_classes_count(self):
        """Verify exactly 10 canonical fruit categories are defined."""
        self.assertEqual(len(CANONICAL_CLASSES), 10)
        expected = [
            'apple', 'avocado', 'banana', 'cherry', 'kiwi',
            'mango', 'orange', 'pineapple', 'strawberry', 'watermelon'
        ]
        self.assertEqual(CANONICAL_CLASSES, expected)

    def test_typo_normalization(self):
        """Verify known dataset typos are mapped to canonical names."""
        self.assertEqual(CANONICAL_CLASS_MAP['pinenapple'], 'pineapple')
        self.assertEqual(CANONICAL_CLASS_MAP['pineapple'], 'pineapple')
        self.assertEqual(CANONICAL_CLASS_MAP['stawberries'], 'strawberry')
        self.assertEqual(CANONICAL_CLASS_MAP['strawberries'], 'strawberry')

    def test_bijective_class_indexing(self):
        """Verify class_to_idx and idx_to_class mappings are bijective."""
        for name, idx in CLASS_TO_IDX.items():
            self.assertEqual(IDX_TO_CLASS[idx], name)

    def test_transforms_construction(self):
        """Verify image transformation pipeline construction."""
        t_eval = get_default_transforms(input_size=(224, 224), is_train=False, augment=False)
        self.assertIsInstance(t_eval, transforms.Compose)
        self.assertEqual(len(t_eval.transforms), 3)

        t_train = get_default_transforms(input_size=(224, 224), is_train=True, augment=True)
        self.assertIsInstance(t_train, transforms.Compose)
        self.assertGreater(len(t_train.transforms), 3)


if __name__ == '__main__':
    unittest.main()
