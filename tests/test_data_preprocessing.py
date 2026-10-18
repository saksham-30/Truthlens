import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import pandas as pd

from src.data_preprocessing import (
    load_and_clean_data,
    save_processed_data,
    stratified_split,
    validate_split_integrity,
    validate_training_manifest,
    write_training_manifest,
)
from src.sample_data_generator import SampleDataGenerator


class GroupedSplitTests(unittest.TestCase):
    @staticmethod
    def make_dataset():
        rows = []
        topics = ('technology', 'education')
        generator = SampleDataGenerator()
        for label, templates in ((0, generator.human_templates), (1, generator.ai_templates)):
            for template in templates:
                for topic in topics:
                    rows.append({
                        'text': template.format(topic) if label == 0 else template.format(topic, topic),
                        'label': label,
                    })
        return pd.DataFrame(rows)

    def test_grouped_split_is_deterministic_and_has_both_labels(self):
        data = self.make_dataset()
        first = stratified_split(data)
        second = stratified_split(data)

        for left, right in zip(first, second):
            self.assertEqual(
                sorted(zip(left['text'], left['split_group_id'])),
                sorted(zip(right['text'], right['split_group_id'])),
            )
            self.assertEqual(set(left['label']), {0, 1})

        validate_split_integrity(*first)

    def test_exact_duplicate_cannot_cross_splits(self):
        train, val, test = self._split()
        train_text = train.loc[train['label'] == 0, 'text'].iloc[0]
        val_index = val.index[val['label'] == 0][0]
        val.loc[val_index, 'text'] = train_text

        with self.assertRaisesRegex(ValueError, 'Exact duplicate text leaks'):
            validate_split_integrity(train, val, test)

    def test_normalized_duplicate_cannot_cross_splits(self):
        train, val, test = self._split()
        train_text = train.loc[train['label'] == 0, 'text'].iloc[0]
        val_index = val.index[val['label'] == 0][0]
        val.loc[val_index, 'text'] = f'  {train_text.upper()}  '

        with self.assertRaisesRegex(ValueError, 'Normalized duplicate text leaks'):
            validate_split_integrity(train, val, test)

    def test_source_group_cannot_cross_splits(self):
        train, val, test = self._split()
        train = train.copy()
        val = val.copy()
        train['source_id'] = None
        val['source_id'] = None
        train_index = train.index[train['label'] == 0][0]
        val_index = val.index[val['label'] == 0][0]
        train.loc[train_index, 'source_id'] = 'shared-source'
        val.loc[val_index, 'source_id'] = 'shared-source'

        with self.assertRaisesRegex(ValueError, 'Recognized template/source groups leak'):
            validate_split_integrity(train, val, test)

    def test_missing_class_is_rejected(self):
        train, val, test = self._split()
        test = test[test['label'] == 0]

        with self.assertRaisesRegex(ValueError, 'test split must contain both labels'):
            validate_split_integrity(train, val, test)

        with self.assertRaisesRegex(ValueError, 'Dataset must contain both labels'):
            stratified_split(self.make_dataset().query('label == 0'))

    def test_missing_group_metadata_is_rejected(self):
        rows = [
            {'text': f'Untracked source sample {label}-{index}.', 'label': label}
            for label in (0, 1)
            for index in range(5)
        ]

        with self.assertRaisesRegex(ValueError, 'group metadata is unavailable'):
            stratified_split(pd.DataFrame(rows))

    def test_invalid_split_is_rejected_before_any_output_is_created(self):
        train, val, test = self._split()
        test = test[test['label'] == 0]

        with TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory) / 'processed'
            with self.assertRaisesRegex(ValueError, 'test split must contain both labels'):
                save_processed_data(train, val, test, str(output_directory))
            self.assertFalse(output_directory.exists())

    def test_cleaning_preserves_source_groups_bridged_by_duplicate_text(self):
        duplicated_text = 'A shared source passage with sufficient length.'
        raw = pd.DataFrame([
            {'text': duplicated_text, 'label': 0, 'source_id': 'source-a'},
            {'text': duplicated_text, 'label': 0, 'source_id': 'source-b'},
            {'text': 'A different passage associated with source a.', 'label': 0, 'source_id': 'source-a'},
            {'text': 'Another distinct passage from source b.', 'label': 0, 'source_id': 'source-b'},
            {'text': 'An AI passage with sufficient example length.', 'label': 1, 'source_id': 'source-ai'},
        ])

        with TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / 'source_data.csv'
            raw.to_csv(input_path, index=False)
            cleaned = load_and_clean_data(str(input_path))

        source_a_group = cleaned.loc[cleaned['source_id'] == 'source-a', 'split_group_id'].iloc[0]
        source_b_group = cleaned.loc[cleaned['source_id'] == 'source-b', 'split_group_id'].iloc[0]
        self.assertEqual(source_a_group, source_b_group)

    def test_training_manifest_matches_exact_training_csv(self):
        with TemporaryDirectory() as temporary_directory:
            train_path = Path(temporary_directory) / 'train.csv'
            manifest_path = Path(temporary_directory) / 'model_manifest.json'
            pd.DataFrame([
                {'text': 'training sample technology', 'label': 0, 'split_group_id': 'group-a'},
                {'text': 'training sample education', 'label': 1, 'split_group_id': 'group-b'},
            ]).to_csv(train_path, index=False)
            write_training_manifest(str(train_path), str(manifest_path), 'test-model')
            validate_training_manifest(str(train_path), str(manifest_path), 'test-model')

            with train_path.open('a', encoding='utf-8') as train_file:
                train_file.write('changed training row,0,group-c\n')
            with self.assertRaisesRegex(ValueError, 'does not match'):
                validate_training_manifest(str(train_path), str(manifest_path), 'test-model')

    def test_missing_training_manifest_is_rejected(self):
        with TemporaryDirectory() as temporary_directory:
            train_path = Path(temporary_directory) / 'train.csv'
            train_path.write_text('text,label,split_group_id\na,0,group-a\nb,1,group-b\n', encoding='utf-8')
            manifest_path = Path(temporary_directory) / 'missing_manifest.json'

            with self.assertRaisesRegex(ValueError, 'provenance is unavailable'):
                validate_training_manifest(str(train_path), str(manifest_path), 'test-model')

    def test_generator_marks_demo_rows_as_synthetic(self):
        generated = SampleDataGenerator().generate_dataset(num_samples=20)
        self.assertEqual(set(generated['dataset_type']), {'synthetic_demo'})

    def _split(self):
        return tuple(frame.copy() for frame in stratified_split(self.make_dataset()))


if __name__ == '__main__':
    unittest.main()