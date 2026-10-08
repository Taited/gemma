import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

script = Path(__file__).resolve().parents[1] / 'scripts/gemma31b_edited_frame_consistency.py'
spec = importlib.util.spec_from_file_location('consistency', script)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


class UnifiedWorklistTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.sources, manifests = [], []
        for i, (sample, anchor) in enumerate([('a', True), ('a', False), ('b', True)]):
            edited = self.root / 'edited' / f'{i}.jpg'
            crop = self.root / 'reference_inputs' / f'{i}.jpg'
            for p in [edited, crop]:
                p.parent.mkdir(exist_ok=True)
                p.touch()
            self.sources.append(dict(job_name=str(i), sample_id=sample, entity_id='primary_object',
                                     frame_index=i, is_anchor=anchor, object_name='cup'))
            manifests.append(dict(source_job_name=str(i), output_path=str(edited), model_input_path=str(crop)))
        self.jobs = self.root / 'jobs.jsonl'
        self.write(self.jobs, self.sources)
        self.write(self.root / 'manifest.shard_00.jsonl', manifests)

    def write(self, path, rows):
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))

    def test_unified_groups_and_peer_isolation(self):
        rows, sources, generated, groups = review.build_worklist(
            SimpleNamespace(jobs=self.jobs, generation_root=self.root))
        self.assertEqual((len(rows), sources, generated, groups), (3, 3, 3, 2))
        self.assertEqual([len(r['peer_edited_paths']) for r in rows], [1, 1, 0])
        self.assertEqual([r['view_type'] for r in rows], ['anchor', 'extra', 'anchor'])

    def test_legacy_matches_unified(self):
        main, extra = self.root / 'main.jsonl', self.root / 'extra.jsonl'
        self.write(main, [r for r in self.sources if r['is_anchor']])
        self.write(extra, [r for r in self.sources if not r['is_anchor']])
        legacy = review.build_worklist(SimpleNamespace(main_jobs=main, extra_jobs=extra,
                                                       main_root=self.root, extra_root=self.root))
        unified = review.build_worklist(SimpleNamespace(jobs=self.jobs, generation_root=self.root))
        self.assertEqual(legacy, unified)

    def test_paired_arguments_required(self):
        with self.assertRaisesRegex(ValueError, 'together'):
            review.build_worklist(SimpleNamespace(jobs=self.jobs))

    def test_duplicate_jobs_fail(self):
        self.write(self.jobs, self.sources + self.sources[:1])
        with self.assertRaisesRegex(ValueError, 'Duplicate source'):
            review.build_worklist(SimpleNamespace(jobs=self.jobs, generation_root=self.root))


if __name__ == '__main__':
    unittest.main()
