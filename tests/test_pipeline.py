import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from jrs_rem.io import RunStore, atomic_json, group_key, sample_key, read_json
from jrs_rem.directions import center, projection, estimate, load_direction, select_calibration
from jrs_rem.evaluation import JUDGE_FILES, labels, summarize, judge
from jrs_rem.analysis import analyze
from jrs_rem.data import prepare
from jrs_rem.probe import group_split
from jrs_rem.utility import eval_mme, eval_scienceqa


def sample(i, variant='SD', dataset='HADES'):
    return dict(dataset=dataset, idx=i, **{'class':'Violence'}, data_type=f'utxt+uimg[{variant}]', txt='Synthetic test prompt', img='synthetic.png')


def fixture(root):
    rows=[sample(i, v) for i in range(8) for v in ('SD','SD_typo')]
    atomic_json(root/'results.json',[dict(r, output_txt='Synthetic answer for test purposes only') for r in rows])
    values=np.random.default_rng(0).normal(size=(len(rows),3,4)).astype(np.float32)
    for i,r in enumerate(rows):values[i,:,0]+=(-4 if r['idx']%2 else 4)
    values[:,0]=0
    np.save(root/'hs.npy',values)
    np.save(root/'text_hs.npy',np.zeros_like(values))
    for name,filename in JUDGE_FILES.items():
        records=[]
        for row in rows:
            safe=bool(row['idx']%2)
            records.append({'key':sample_key(row),'is_safe':safe,'error':('Safe' if safe else 'Unsafe') if name.startswith('Qwen') else None})
        atomic_json(root/filename,records)
    return rows


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()

    def test_center_uses_arithmetic_mean(self):
        self.assertEqual(float(center(np.array([[0.],[2.],[4.],[8.]]))[0]),3.5)

    def test_projection_preserves_signed_coordinates(self):
        ds=np.array([[1.,-1.]])/np.sqrt(2)
        cos,vector=projection(ds,ds)
        np.testing.assert_allclose(cos,1,atol=1e-6)
        np.testing.assert_allclose(vector,ds,atol=1e-6)
        self.assertLess(vector[0,1],0)
        with self.assertRaises(ValueError):projection(ds,ds,'legacy_nonnegative')

    def test_zero_shift_finite(self):
        c,v=projection(np.zeros((2,4)),np.ones((2,4)))
        self.assertTrue(np.isfinite(c).all());self.assertTrue(np.equal(v,0).all())

    def test_prompt_variants_and_implicit_alias_share_group(self):
        self.assertEqual(group_key(sample(1,'SD')),group_key(sample(1,'TYPO')))
        self.assertEqual(group_key(sample(1)),group_key(sample(1,dataset='HADES_hideharm')))
        self.assertNotEqual(sample_key(sample(1,'SD')),sample_key(sample(1,'TYPO')))

    def test_disjoint_calibration_and_saved_exclusion(self):
        rows=fixture(self.root)
        info=estimate(self.root,self.root/'d.npz',count=2)
        self.assertEqual(len(info['calibration_groups']),4)
        self.assertEqual(info['evaluation_count'],8)
        ds,meta=load_direction(self.root/'d.npz')
        self.assertEqual(ds.shape,(3,4))
        with np.load(self.root/'d.npz') as artifact:
            self.assertEqual(set(artifact.files), {'ds', 'metadata'})
        self.assertEqual(info,meta)
        self.assertEqual(meta['center'], 'mean')
        a_idx = [i for i, r in enumerate(rows) if sample_key(r) in meta['refusal_samples']]
        b_idx = [i for i, r in enumerate(rows) if sample_key(r) in meta['jailbreak_samples']]
        hs = np.load(self.root/'hs.npy')
        expected = hs[b_idx].mean(axis=0) - hs[a_idx].mean(axis=0)
        expected /= np.maximum(np.linalg.norm(expected, axis=-1, keepdims=True), 1e-8)
        np.testing.assert_allclose(ds, expected, atol=1e-6)
        safe=labels(self.root,'unanimous')
        a,b=select_calibration(rows,safe,2,0)
        self.assertTrue({group_key(rows[i]) for i in a}.isdisjoint({group_key(rows[i]) for i in b}))

    def test_insufficient_unique_prompts_fails(self):
        rows=[sample(0,v) for v in ('SD','TYPO')]+[sample(1,v) for v in ('SD','TYPO')]
        with self.assertRaises(ValueError):select_calibration(rows,[1,1,0,0],2,0)

    def test_majority_and_unanimity_differ(self):
        fixture(self.root)
        p=self.root/JUDGE_FILES['machine'];r=read_json(p);r[0]['is_safe']=True;atomic_json(p,r)
        self.assertEqual(labels(self.root)[0],0)
        self.assertEqual(labels(self.root,'unanimous')[0],-1)

    def test_controversial_is_unsafe(self):
        fixture(self.root)
        p=self.root/JUDGE_FILES['Qwen3Guard-Gen-8B'];r=read_json(p)
        r[0]['error']='Controversial';r[0]['is_safe']=True;atomic_json(p,r)
        self.assertEqual(labels(self.root,'unanimous')[0],0)

    def test_malformed_qwen_cannot_become_safe(self):
        fixture(self.root)
        p=self.root/JUDGE_FILES['Qwen3Guard-Gen-8B'];r=read_json(p);r[0]['error']=None;atomic_json(p,r)
        with self.assertRaises(ValueError):labels(self.root)

    def test_permuted_judge_ids_fail(self):
        fixture(self.root)
        p=self.root/JUDGE_FILES['machine'];r=read_json(p);r.reverse();atomic_json(p,r)
        with self.assertRaises(ValueError):labels(self.root)

    def test_asr_counts(self):
        fixture(self.root);result=summarize(self.root)
        self.assertEqual(result['HADES/all/overall'],{'unsafe':8,'count':16,'asr':.5})

    def test_machine_judge_rejects_stale_result_content(self):
        rows=[dict(sample(1),output_txt="I cannot comply with this request.")]
        atomic_json(self.root/'results.json',rows);judge(self.root,'machine')
        rows[0]['output_txt']='A different synthetic answer';atomic_json(self.root/'results.json',rows)
        with self.assertRaises(ValueError):judge(self.root,'machine')

    def test_resume_records_and_hidden_alignment(self):
        rows=[sample(1),sample(2)];cfg={'save_hidden':True}
        store=RunStore(self.root,rows,cfg)
        h=np.ones((3,4));store.save(rows[0],'a',h,h*2)
        self.assertTrue(store.done(rows[0]));self.assertFalse(store.done(rows[1]))
        with self.assertRaises(ValueError):store.finalize()
        resumed=RunStore(self.root,rows,cfg);resumed.save(rows[1],'b',h*3,h*4)
        records=resumed.finalize()
        self.assertEqual([r['output_txt'] for r in records],['a','b'])
        np.testing.assert_array_equal(np.load(self.root/'hs.npy')[:,0,0],[1,3])
        np.testing.assert_array_equal(np.load(self.root/'text_hs.npy')[:,0,0],[2,4])

    def test_resume_different_config_or_sample_order_rejected(self):
        rows=[sample(1),sample(2)];RunStore(self.root,rows,{'save_hidden':False})
        with self.assertRaises(ValueError):RunStore(self.root,rows[::-1],{'save_hidden':False})
        with self.assertRaises(ValueError):RunStore(self.root,rows,{'save_hidden':True})

    def test_analysis_outputs_with_paired_shifts(self):
        fixture(self.root);estimate(self.root,self.root/'d.npz',count=2)
        hs=np.load(self.root/'hs.npy');hs[:,0]=1;np.save(self.root/'hs.npy',hs)
        analyze([self.root],self.root/'analysis',layer=1,direction=self.root/'d.npz')
        self.assertTrue((self.root/'analysis/representations.csv').exists())
        self.assertEqual(read_json(self.root/'analysis/shift/shift.json')['paired_samples'],16)
        with self.assertRaises(FileExistsError):analyze([self.root],self.root/'analysis',layer=1)

    def test_hades_loader_explicit_image_step(self):
        base=self.root/'HADES'
        for category in ['Animal','Financial','Privacy','Self-Harm','Violence']:
            atomic_json(base/'instructions'/f'{category}.json',[{'id':0,'instruction':'Synthetic test'}])
            image=base/'optimized_SD_images'/category/'0';image.mkdir(parents=True)
            for n in ['opt_5.jpg','concat_opt_5.jpg','keyword.jpg']:(image/n).write_bytes(b'test')
        with self.assertRaises(FileNotFoundError):prepare('HADES',self.root,self.root/'default.json')
        rows=prepare('HADES',self.root,self.root/'step5.json',hades_step=5)
        self.assertEqual(len(rows),15);self.assertTrue(any('opt_5.jpg' in r['img'] for r in rows))

    def test_mme_pair_scoring_and_missing_gt_rejected(self):
        gt=[{'question_id':1,'answer':'yes'},{'question_id':2,'answer':'no'}]
        atomic_json(self.root/'MME/MME.json',gt)
        rows=[{'idx':1,'class':'existence','img':'a','output_txt':'Yes.'},
              {'idx':2,'class':'existence','img':'a','output_txt':'No.'}]
        self.assertEqual(eval_mme(rows,self.root)[0],200)
        rows[1]['idx']=999
        with self.assertRaises(ValueError):eval_mme(rows,self.root)

    def test_scienceqa_original_answer_priority(self):
        from jrs_rem.utility import parse_scienceqa_answer
        choices = ['red', 'blue', 'green']
        cases = {'B': 1, 'B. red': 1, '(B) red': 1, 'blue': 1,
                 'The answer is A, blue': 1, 'blue and red': 0,
                 'Selected option C': 2, 'No answer': -1, 'E': -1}
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(parse_scienceqa_answer(text, choices), expected)
        atomic_json(self.root/'ScienceQA/test_data.json',
                    [{'idx': 1, 'choices': choices, 'answer': 1},
                     {'idx': 2, 'choices': choices, 'answer': 0}])
        rows = [{'idx': 1, 'data_type': 'stxt+rimg', 'output_txt': 'blue'},
                {'idx': 2, 'data_type': 'stxt+rimg', 'output_txt': 'No answer'}]
        self.assertEqual(eval_scienceqa(rows, self.root), (.5, {'stxt+rimg': .5}))

    def test_group_split_has_no_prompt_variant_leakage(self):
        rows=[sample(i,v) for i in range(20) for v in ('SD','TYPO')]
        mask=group_split(rows,0)
        self.assertTrue(all(mask[i]==mask[i+1] for i in range(0,len(rows),2)))


if __name__=='__main__':unittest.main()
