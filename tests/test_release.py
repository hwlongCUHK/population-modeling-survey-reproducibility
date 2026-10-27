"""Offline tests; synthetic examples do not represent empirical study results."""
import itertools
import json
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from population_survey.grouping import assignment,balanced_kmeans
from population_survey.metrics import js,paired
from population_survey.prompts import prompt,strict_parse
from reproduce_statistics import reproduce
from verify_release import verify

class ReleaseTests(unittest.TestCase):
    def test_public_statistics(self):
        self.assertTrue(reproduce()['status'].startswith('PASS'))

    def test_private_data_exclusion_and_hashes(self):
        verify()

    def test_fixed_selections_and_group_sizes(self):
        selections=json.loads((ROOT/'results/matched_controls/fixed_selections.json').read_text())
        self.assertEqual(len(selections),13)
        real=next(iter(selections.values()))['llm']['real'];pool=set(sum(real,[]));self.assertEqual(len(pool),80)
        for q,s in selections.items():
            self.assertEqual(s['llm']['real'],real)
            for arm,ids in s['ssr'].items():self.assertEqual(len(ids),500);self.assertEqual(len(set(ids)),500)
            for arm,groups in s['llm'].items():
                self.assertEqual(len(groups),10);self.assertTrue(all(len(g)==8 for g in groups))
                if arm.startswith('random_'):self.assertEqual(set(sum(groups,[])),pool)
        km=json.loads((ROOT/'results/kmeans/group_assignments.json').read_text())
        self.assertEqual(km['real'],real)
        self.assertEqual(set(sum(km['kmeans_balanced'],[])),pool)
        self.assertTrue(all(len(g)==8 for g in km['kmeans_balanced']))
        self.assertNotEqual({frozenset(g) for g in real},{frozenset(g) for g in km['kmeans_balanced']})

    def test_raw_output_aggregation_matches_original_real_arm(self):
        def read(label):return {r['job_id']:r for r in [json.loads(l) for l in (ROOT/f'results/{label}/numeric_model_outputs.jsonl').read_text().splitlines()]}
        old=read('matched_controls');new=read('kmeans')
        real=[r for r in new.values() if r['arm']=='real'];self.assertEqual(len(real),130)
        for r in real:self.assertEqual(r['output'],old[r['job_id']]['output'])

    def test_assignment_matches_exhaustive_optimum(self):
        rng=np.random.default_rng(42)
        for n in range(1,7):
            for _ in range(4):
                c=rng.normal(size=(n,n));a=assignment(c)
                optimum=min(sum(c[i,p[i]] for i in range(n)) for p in itertools.permutations(range(n)))
                self.assertEqual(len(set(a)),n);self.assertAlmostEqual(c[np.arange(n),a].sum(),optimum,places=9)

    def test_capacity_and_determinism_on_synthetic_features(self):
        x=np.repeat(np.eye(10),8,axis=0)
        a,s=balanced_kmeans(x,restarts=2);b,t=balanced_kmeans(x,restarts=2)
        np.testing.assert_array_equal(a,b);self.assertEqual(s,t)
        np.testing.assert_array_equal(np.bincount(a),[8]*10)
        for r in s['runs']:self.assertTrue(np.all(np.diff(r['trace'])<=1e-8))

    def test_parsing_metrics_and_prompt(self):
        self.assertEqual(strict_parse('{"distribution":[2,2]}',2),[.5,.5])
        for s in ['[0,0]','[NaN,1]','[true,1]','[-1,2]','[1]','explanation [1,1]']:self.assertIsNone(strict_parse(s,2))
        self.assertAlmostEqual(js([1,0],[0,1]),np.log(2));self.assertEqual(js([1,2],[2,4]),0)
        stats=paired({'q':0.2},{'q':0.2},['q']);self.assertEqual(stats['difference_a_minus_b'],0);self.assertEqual(stats['exact_two_sided_signflip_p'],1)
        p=prompt({'question':'SYNTHETIC QUESTION ONLY','options':['SYNTHETIC OPTION A','SYNTHETIC OPTION B']},['SYNTHETIC POST ONLY'])
        self.assertIn('SYNTHETIC QUESTION ONLY',p);self.assertIn('SYNTHETIC POST ONLY',p)

    def test_private_adapters_with_synthetic_inputs(self):
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC_RELEASE_TEST_ONLY_') as td:
            t=Path(td);index=json.loads((ROOT/'results/question_index.json').read_text())
            groups=json.loads((ROOT/'results/kmeans/group_assignments.json').read_text())['real'];ids=sum(groups,[])
            data={'questions':[{'key':q['id'],'question':'SYNTHETIC TEST QUESTION','options':[f'SYNTHETIC OPTION {i}' for i in range(q['n_options'])]} for q in index],
                  'posts':[{'id':i,'text':'SYNTHETIC TEST POST ONLY'} for i in ids],'real_groups':groups}
            (t/'input.json').write_text(json.dumps(data))
            np.savez(t/'features.npz',ids=np.array(ids),vectors=np.repeat(np.eye(10),8,axis=0))
            cmd=[sys.executable,str(ROOT/'scripts/prepare_private_kmeans.py'),'--input',str(t/'input.json'),
                 '--features-npz',str(t/'features.npz'),'--feature-description','SYNTHETIC TEST ONLY','--out',str(t/'run')]
            r=subprocess.run(cmd,capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr)
            run=t/'run';jobs=json.loads((run/'jobs.json').read_text());self.assertEqual(len(jobs),260)
            digest=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
            (run/'inference_provenance.json').write_text(json.dumps({'jobs_sha256':digest(run/'jobs.json'),'grouping_sha256':digest(run/'grouping_provenance.json'),'synthetic_test_only':True}))
            outputs=[]
            for j in jobs:
                raw=json.dumps({'distribution':[1]*j['n_options']});d={k:j[k] for k in ['job_id','key','arm','group']}
                d.update(output=raw,prediction=strict_parse(raw,j['n_options']),valid=True,input_tokens=1);outputs.append(d)
            (run/'llm_outputs.jsonl').write_text(''.join(json.dumps(d)+'\n' for d in outputs))
            (t/'targets.json').write_text(json.dumps({q['id']:[1]*q['n_options'] for q in index}))
            r=subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_private_kmeans.py'),'--run-dir',str(run),'--targets',str(t/'targets.json')],capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)
            summary=json.loads((run/'evaluation.json').read_text())['primary_all13']
            self.assertEqual(summary['n'],13);self.assertEqual(summary['difference_kmeans_minus_real'],0)

if __name__=='__main__':unittest.main(verbosity=2)
