"""Recompute reported paired statistics from public, anonymous question-level scores."""
import argparse
import json
import math
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from population_survey.metrics import paired
from population_survey.prompts import strict_parse

def load(p):return json.loads(p.read_text())

def assert_same(actual,expected,path='root'):
    if isinstance(expected,dict):
        assert set(actual)==set(expected),(path,'keys differ')
        for k in expected:assert_same(actual[k],expected[k],path+'.'+k)
    elif isinstance(expected,list):
        assert len(actual)==len(expected),(path,'length differs')
        for i,(a,b) in enumerate(zip(actual,expected)):assert_same(a,b,f'{path}[{i}]')
    elif type(expected) in (int,float):
        assert math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-12),(path,actual,expected)
    else:assert actual==expected,(path,actual,expected)

def verify_outputs(directory,questions):
    rows=[json.loads(l) for l in (directory/'numeric_model_outputs.jsonl').read_text().splitlines()]
    seen=set();invalid=[]
    for row in rows:
        assert row['job_id'] not in seen,'Duplicate model output';seen.add(row['job_id'])
        assert row['job_id']==f'{row["key"]}|{row["arm"]}|{row["group"]}'
        prediction=strict_parse(row['output'],questions[row['key']]['n_options'])
        assert prediction==row['prediction']
        assert row['valid']==(prediction is not None)
        if prediction is None:invalid.append(row['job_id'])
    return {'jobs':len(rows),'invalid_jobs':invalid}

def matched_statistics(question_index):
    directory=ROOT/'results/matched_controls'
    rows=load(directory/'per_question_scores.json');keys=[q['id'] for q in question_index]
    original=[q['id'] for q in sorted(question_index,key=lambda q:q['original_six_rank'] or 99) if q['original_six']]
    assert set(rows)==set(keys)
    scores={arm:{k:v[arm] for k,v in rows.items()} for arm in next(iter(rows.values()))}
    comps=[('SSR/direct_clustered','SSR/cluster_selected'),('SSR/direct_full','SSR/cluster_selected'),
           ('SSR/cluster_selected','SSR/random_clustered'),('LLM/real','LLM/random_mean_JS'),
           ('LLM/direct_clustered','LLM/real'),('LLM/direct_full','LLM/real'),
           ('LLM/direct_clustered','LLM/random_101'),('LLM/direct_full','LLM/random_101')]
    comps += [('LLM/real',f'LLM/random_{s}') for s in [101,202,303,404,505]]
    comparisons={subset:{f'{a} vs {b}':paired(scores[a],scores[b],kk) for a,b in comps}
                 for subset,kk in [('all13',keys),('original6',original)]}
    primary=['SSR/direct_clustered vs SSR/cluster_selected','SSR/direct_full vs SSR/cluster_selected','LLM/real vs LLM/random_mean_JS']
    vals=[(k,comparisons['all13'][k]) for k in primary];running=0
    for i,(name,v) in enumerate(sorted(vals,key=lambda kv:kv[1]['exact_two_sided_signflip_p'])):
        running=max(running,min(1,v['exact_two_sided_signflip_p']*(len(vals)-i)))
        v['holm_p_primary_family']=running
    summary={}
    for subset,kk in [('all13',keys),('original6',original)]:
        summary[subset]={}
        for arm,v in scores.items():
            vv=[v[k] for k in kk if v[k] is not None]
            summary[subset][arm]={'n':len(vv),'mean_js':float(np.mean(vv)) if vv else None}
    ssr=load(directory/'semantic_predictions.json')
    return {'summary':summary,'comparisons':comparisons,
            'ssr_grouping_invariance_max_abs':max(v['grouping_invariance_max_abs'] for v in ssr.values())}

def kmeans_statistics(question_index):
    rows=load(ROOT/'results/kmeans/per_question_scores.json');keys=[q['id'] for q in question_index]
    original=[q['id'] for q in sorted(question_index,key=lambda q:q['original_six_rank'] or 99) if q['original_six']]
    assert set(rows)==set(keys)
    def subset(kk):
        a={k:rows[k]['kmeans_balanced'] for k in kk};b={k:rows[k]['real'] for k in kk}
        s=paired(a,b,kk)
        return {'n':s['n'],'keys':s['keys'],'excluded_keys':[k for k in kk if k not in s['keys']],
                'mean_js_kmeans':s['mean_a'],'mean_js_real':s['mean_b'],
                'difference_kmeans_minus_real':s['difference_a_minus_b'],
                'exact_two_sided_signflip_p_raw':s['exact_two_sided_signflip_p'],
                'descriptive_question_bootstrap_ci95':s['ci95_question_bootstrap'],
                'kmeans_better_questions':s['a_better'],'kmeans_worse_questions':s['b_better'],
                'exact_ties':sum(v==0 for v in s['per_question_differences'].values()),
                'per_question_difference':s['per_question_differences']}
    return {'primary_all13':subset(keys),'secondary_original6':subset(original),
            'jobs_total':260,'jobs_missing':0,'jobs_invalid':0}

def reproduce():
    index=load(ROOT/'results/question_index.json');byid={q['id']:q for q in index}
    matched=matched_statistics(index);km=kmeans_statistics(index)
    assert_same(matched,load(ROOT/'results/matched_controls/reference_statistics.json'))
    assert_same(km,load(ROOT/'results/kmeans/reference_statistics.json'))
    checks={label:verify_outputs(ROOT/'results'/label,byid) for label in ['matched_controls','kmeans']}
    assert checks['matched_controls']['jobs']==1040 and len(checks['matched_controls']['invalid_jobs'])==1
    assert checks['kmeans']['jobs']==260 and not checks['kmeans']['invalid_jobs']
    return {'status':'PASS: all published statistics reproduced','output_checks':checks,'matched_controls':matched,'kmeans':km}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path)
    args=ap.parse_args();report=reproduce()
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(report['status'])
    print('Numeric model responses: 1040 matched-control outputs (1 invalid) and 260 K-means-control outputs (0 invalid).')
    s=report['kmeans']['primary_all13']
    print(f'K-means - real: {s["difference_kmeans_minus_real"]:+.8f}; raw two-sided p={s["exact_two_sided_signflip_p_raw"]:.8f}; n={s["n"]}.')
    print('Survey targets are excluded: this command reproduces statistical summaries from the released per-question JS scores.')

if __name__=='__main__':main()
