import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path
import numpy as np
from jrs_rem.analysis import analyze, cosine_distances
from jrs_rem.io import atomic_json, read_json, sample_key
from jrs_rem.evaluation import JUDGE_FILES
from jrs_rem.shifts import analyze_shifts
from jrs_rem.warnings import analyze_warnings, warning_matches
from jrs_rem.probe import class_metrics, train_three_class


def write_run(root, rows, hs, safe):
    atomic_json(root/'results.json', rows)
    np.save(root/'hs.npy', hs)
    np.save(root/'text_hs.npy', np.zeros_like(hs))
    for name, filename in JUDGE_FILES.items():
        atomic_json(root/filename, [{'key':sample_key(r), 'is_safe':bool(y),
                    'error':('Safe' if y else 'Unsafe') if name.startswith('Qwen') else None}
                   for r,y in zip(rows,safe)])


def row(i, category, variant='SD'):
    return {'dataset':'synthetic', 'class':'test', 'idx':i,
            'data_type': ('stxt+rimg' if category==0 else 'utxt+uimg') + f'[{variant}]',
            'txt':f'prompt {i}', 'img':'synthetic.png', 'output_txt':'Synthetic answer'}


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()

    def test_cosine_mean_not_median_and_all_layers(self):
        hs=np.array([[[1.,0.],[0.,1.]],[[1.,0.],[0.,1.]],[[0.,3.],[3.,0.]],[[0.,1.],[1.,0.]]])
        dist=cosine_distances(hs,['jailbreak']*3+['refusal'])
        # Mean jailbreak direction is (2,3), while the coordinate median is (1,0).
        np.testing.assert_allclose(dist[-1], [1-3/np.sqrt(13)]*2, atol=1e-6)
        with self.assertRaises(ValueError):cosine_distances(np.zeros_like(hs),['jailbreak']*4)

    def test_distance_summary_is_population_std(self):
        rows=[row(i,2 if i<2 else 1) for i in range(3)]
        hs=np.array([[[1.,0.]],[[0.,1.]],[[0.,2.]]],dtype=np.float32)
        write_run(self.root/'input',rows,hs,[0,0,1])
        analyze([self.root/'input'],self.root/'out',layer=0)
        with (self.root/'out/distance_summary.csv').open() as f: summary=list(csv.DictReader(f))
        self.assertEqual(len(summary),2)
        jb=next(r for r in summary if r['label']=='jailbreak')
        self.assertAlmostEqual(float(jb['mean']),1-1/np.sqrt(2),places=6)
        self.assertAlmostEqual(float(jb['std']),0,places=6)

    def test_warning_denominator_and_no_refusal_keyword(self):
        rows=[row(i,2) for i in range(4)]
        rows[0]['output_txt']='This is ILLEGAL. It is harmful.'
        rows[1]['output_txt']='No warning here.'
        rows[2]['output_txt']='Illegal, I refuse.'
        rows[3]['data_type']='stxt+rimg[SD]';rows[3]['output_txt']='Harmful'
        write_run(self.root/'input',rows,np.ones((4,1,2)),[0,0,1,1])
        result=analyze_warnings([self.root/'input'],self.root/'out')
        self.assertEqual(result['groups'][0]['jailbreak_count'],2)
        self.assertEqual(result['groups'][0]['ratio'],.5)
        self.assertEqual(warning_matches("I'm sorry, I cannot help."),[])

    def test_shift_signed_values_and_explicit_text_filter(self):
        rows=[row(0,2),row(1,1)]
        hs=np.array([[[1.,-1.]],[[0.,0.]]],dtype=np.float32)
        write_run(self.root/'input',rows,hs,[0,1])
        import json
        ds=np.array([[1.,-1.]])/np.sqrt(2)
        np.savez(self.root/'d.npz',ds=ds,metadata=json.dumps({'calibration_groups':[]}))
        result=analyze_shifts([self.root/'input'],self.root/'out',self.root/'d.npz')
        self.assertEqual(result['paired_samples'],2)
        with (self.root/'out/shifts.csv').open() as f: values=list(csv.DictReader(f))
        self.assertAlmostEqual(float(values[0]['cosine']),1,places=6)
        self.assertAlmostEqual(float(values[0]['signed_projection']),np.sqrt(2),places=6)
        self.assertEqual(float(values[1]['cosine']),0)
        texts=[dict(r,img=None,data_type='utxt') for r in rows]
        write_run(self.root/'text',texts,hs,[1,0])
        result=analyze_shifts([self.root/'input'],self.root/'filtered',self.root/'d.npz',text_results=[self.root/'text'])
        self.assertEqual(result['paired_samples'],1)
        self.assertEqual(result['excluded_text_unsafe_or_conflict'],1)

    def test_per_class_f1(self):
        metrics=class_metrics(np.array([0,0,1,2]),np.array([0,1,1,1]))
        self.assertAlmostEqual(metrics[0]['f1'],2/3)
        self.assertEqual(metrics[1]['f1'],.5)
        self.assertEqual(metrics[2]['f1'],0)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Requires PyTorch')
    def test_three_class_training_has_disjoint_groups_and_shared_layer_split(self):
        rows=[row(i,i%3,v) for i in range(60) for v in ['SD','TYPO']]
        hs=np.array([[np.eye(3)[r['idx']%3]]*2 for r in rows],dtype=np.float32)
        write_run(self.root/'input',rows,hs,[r['idx']%3!=2 for r in rows])
        result=train_three_class([self.root/'input'],self.root/'out',epochs=1,train_cap=6,test_cap=4)
        self.assertTrue(set(result['train_groups']).isdisjoint(result['test_groups']))
        self.assertEqual(len(result['train_samples']),18)
        self.assertEqual(len(result['test_samples']),12)
        self.assertEqual(len(result['reports']),2)
        self.assertTrue((self.root/'out/probe_layer_1.pt').exists())
        self.assertEqual(set(result['reports'][0]['classes']),{'benign','refusal','jailbreak'})
