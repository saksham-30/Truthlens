import unittest

import pandas as pd

from src.external_pilot import (
    _candidate_directory,
    choose_validation_threshold,
    validate_approval_manifest,
    validate_external_frame,
)
from src.data_preprocessing import stratified_split, validate_split_integrity


def approved_manifest():
    return {
        'dataset_id': 'unit-test-only',
        'revision': 'test-revision',
        'source_url': 'https://example.invalid/source',
        'license': 'test-only',
        'license_review_status': 'approved',
        'training_use_approved': True,
        'approval_reference': 'unit-test-approval',
        'retrieval_date': '2026-10-18',
        'dataset_sha256': 'a' * 64,
        'label_mapping': {'0': 'human', '1': 'generated'},
    }


class ExternalPilotContractTests(unittest.TestCase):
    def test_pending_approval_fails_closed(self):
        manifest = approved_manifest()
        manifest['license_review_status'] = 'pending'
        with self.assertRaisesRegex(ValueError, 'explicitly approved'):
            validate_approval_manifest(manifest)

    def test_missing_label_mapping_fails_closed(self):
        manifest = approved_manifest()
        manifest['label_mapping'] = {'0': 'human'}
        with self.assertRaisesRegex(ValueError, 'label_mapping'):
            validate_approval_manifest(manifest)

    def test_synthetic_rows_are_rejected(self):
        frame = pd.DataFrame([{
            'text': 'A fixture row', 'label': 0, 'group_id': 'g0', 'dataset_type': 'synthetic_demo'
        }])
        with self.assertRaisesRegex(ValueError, 'synthetic demo data is blocked'):
            validate_external_frame(frame, approved_manifest())

    def test_missing_dataset_type_is_rejected(self):
        frame = pd.DataFrame([{
            'text': 'A fixture row', 'label': 0, 'group_id': 'g0', 'dataset_type': None
        }])
        with self.assertRaisesRegex(ValueError, 'dataset_type=external'):
            validate_external_frame(frame, approved_manifest())

    def test_fractional_label_is_rejected(self):
        frame = pd.DataFrame([
            {'text': 'Human fixture', 'label': 0.5, 'group_id': 'g0', 'dataset_type': 'external'},
            {'text': 'AI fixture', 'label': 1, 'group_id': 'g1', 'dataset_type': 'external'},
        ])
        with self.assertRaisesRegex(ValueError, 'valid labels 0 and 1'):
            validate_external_frame(frame, approved_manifest())

    def test_external_contract_splits_groups_without_overlap(self):
        rows = []
        for label in (0, 1):
            for group_number in range(5):
                for sample_number in range(2):
                    rows.append({
                        'text': f'External fixture class {label}, group {group_number}, sample {sample_number}.',
                        'label': label,
                        'group_id': f'class-{label}-group-{group_number}',
                        'dataset_type': 'external',
                    })
        frame = validate_external_frame(pd.DataFrame(rows), approved_manifest())
        train, val, test = stratified_split(frame)
        validate_split_integrity(train, val, test)
        self.assertEqual(set(train['label']), {0, 1})
        self.assertEqual(set(val['label']), {0, 1})
        self.assertEqual(set(test['label']), {0, 1})
        self.assertFalse(set(train['split_group_id']) & set(val['split_group_id']))
        self.assertFalse(set(train['split_group_id']) & set(test['split_group_id']))
        self.assertFalse(set(val['split_group_id']) & set(test['split_group_id']))

    def test_validation_threshold_uses_only_passed_validation_values(self):
        threshold = choose_validation_threshold([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9])
        self.assertEqual(threshold, 0.6)

    def test_production_artifact_path_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'production paths are forbidden'):
            _candidate_directory('models/logistic_regression.pkl')


if __name__ == '__main__':
    unittest.main()