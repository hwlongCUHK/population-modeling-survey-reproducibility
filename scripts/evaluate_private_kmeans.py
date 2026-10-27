"""Introduce survey counts only here, after predictions are frozen."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from population_survey.prompts import strict_parse
from prepare_private_kmeans import digest,save

BASE=Path(__file__).resolve().parents[1]
ORIGINAL6=[q['id'] for q in sorted(json.loads((BASE/'results/question_index.json').read_text()),key=lambda q:q['original_six_rank'] or 99) if q['original_six']]

def js(p,q):
    p=np.asarray(p,dtype=float);q=np.asarray(q,dtype=float)
    if p.ndim!=1 or p.shape!=q.shape or not np.isfinite(p).all() or not np.isfinite(q).all() or min(p.min(),q.min())<0 or min(p.sum(),q.sum())<=0:
        raise ValueError('Invalid probability/count vector')
    p=p/p.sum();q=q/q.sum();m=(p+q)/2
    return float(sum(np.sum(x[x>0]*np.log(x[x>0]/m[x>0])) for x in [p,q])/2)

def paired(rows,keys):
    complete=[k for k in keys if rows[k]['real']['js'] is not None and rows[k]['kmeans_balanced']['js'] is not None]
    if not complete:return {'n':0,'excluded_keys':keys}
    a=np.array([rows[k]['kmeans_balanced']['js'] for k in complete]);b=np.array([rows[k]['real']['js'] for k in complete]);d=a-b
    null=np.array([np.mean(d*np.array(s)) for s in itertools.product([-1,1],repeat=len(d))])
    boot=np.random.default_rng(20261026).choice(d,size=(20000,len(d)),replace=True).mean(axis=1)
    return {'n':len(d),'keys':complete,'excluded_keys':[k for k in keys if k not in complete],
            'mean_js_kmeans':float(a.mean()),'mean_js_real':float(b.mean()),'difference_kmeans_minus_real':float(d.mean()),
            'exact_two_sided_signflip_p_raw':float(np.mean(abs(null)>=abs(d.mean())-1e-14)),
            'descriptive_question_bootstrap_ci95':np.quantile(boot,[.025,.975]).tolist(),
            'kmeans_better_questions':int((d<0).sum()),'kmeans_worse_questions':int((d>0).sum()),
            'exact_ties':int((d==0).sum()),'per_question_difference':dict(zip(complete,d.tolist()))}

def evaluate(jobs,outputs,targets):
    byid={j['job_id']:j for j in jobs}
    if len(byid)!=len(jobs):raise ValueError('Duplicate jobs')
    done={}
    for d in outputs:
        jid=d['job_id']
        if jid not in byid or jid in done:raise ValueError('Unknown/duplicate model output')
        j=byid[jid]
        if any(j[k]!=d[k] for k in ['key','arm','group']):raise ValueError('Output metadata differs from job')
        pred=strict_parse(d['output'],j['n_options'])
        if pred!=d['prediction'] or d['valid']!=(pred is not None):raise ValueError('Raw output / stored prediction mismatch')
        done[jid]=d
    rows={}
    for k,t in targets.items():
        rows[k]={}
        for arm in ['real','kmeans_balanced']:
            jj=[j for j in jobs if j['key']==k and j['arm']==arm]
            if len(jj)!=10:raise ValueError('Expected ten jobs per question-arm')
            missing=[j['job_id'] for j in jj if j['job_id'] not in done]
            invalid=[j['job_id'] for j in jj if j['job_id'] in done and not done[j['job_id']]['valid']]
            dd=[done[j['job_id']] for j in jj if j['job_id'] in done]
            pred=np.mean([d['prediction'] for d in dd],axis=0).tolist() if not missing and not invalid else None
            rows[k][arm]={'prediction':pred,'js':js(pred,t) if pred is not None else None,
                          'valid_groups':sum(d['valid'] for d in dd),'missing_job_ids':missing,'invalid_job_ids':invalid,
                          'input_tokens':sum(d['input_tokens'] for d in dd)}
    if {j['key'] for j in jobs}!=set(targets):raise ValueError('Target question keys differ from jobs')
    return rows

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--targets',type=Path,required=True)
    args=ap.parse_args();out=args.run_dir.resolve()
    if out.is_relative_to(BASE): raise SystemExit('Private run directory must be outside the repository')
    grouping=json.loads((out/'grouping_provenance.json').read_text());prov=json.loads((out/'inference_provenance.json').read_text())
    if digest(out/'jobs.json')!=grouping['jobs_sha256'] or digest(out/'jobs.json')!=prov['jobs_sha256'] or digest(out/'grouping_provenance.json')!=prov['grouping_sha256']:
        raise SystemExit('Run provenance differs from frozen inputs')
    jobs=json.loads((out/'jobs.json').read_text());targets=json.loads(args.targets.read_text())
    outputs=[json.loads(l) for l in (out/'llm_outputs.jsonl').read_text().splitlines()]
    rows=evaluate(jobs,outputs,targets);main_stats=paired(rows,list(targets))
    missing=sum(len(v['missing_job_ids']) for q in rows.values() for v in q.values())
    invalid=sum(len(v['invalid_job_ids']) for q in rows.values() for v in q.values())
    report={'primary_all13':main_stats,'secondary_original6':paired(rows,ORIGINAL6),'per_question':rows,
            'jobs_total':len(jobs),'jobs_missing':missing,'jobs_invalid':invalid,
            'targets_sha256':digest(args.targets),'model_outputs_sha256':digest(out/'llm_outputs.jsonl'),
            'notes':['JS divergence uses natural logarithms, not square-root JS distance; lower is better.',
                     'One new planned comparison. Raw exact sign-flip p is not multiplicity-adjusted across previous rebuttal experiments.',
                     'Sign-flip reference assumes exchangeable signs under the null; 13 questions share one domain and evidence pool.',
                     'Bootstrap intervals resample questions and are descriptive, not population or across-domain confidence intervals.',
                     'Any invalid group invalidates the question-arm. No uniform fallback, dropped-group average or best-of-retry.',
                     'This is fixed-evidence grouping sensitivity, not full-corpus population recovery.']}
    save(out/'evaluation.json',report)
    if missing or not main_stats['n']:
        text='RESULTS NOT READY: missing jobs or no complete pairs. Finish inference and inspect evaluation.json. No placeholders filled.\n'
    else:
        s=main_stats;ci=s['descriptive_question_bootstrap_ci95']
        finding=(f'With the same posts and inference budget, capacity-constrained K-means yielded JS={s["mean_js_kmeans"]:.4f} versus {s["mean_js_real"]:.4f} for the original grouping, '
                 f'with lower/higher JS on {s["kmeans_better_questions"]}/{s["kmeans_worse_questions"]} questions ({s["exact_ties"]} ties). '
                 'This quantifies question-level sensitivity to evidence organization.')
        text=(f'[KM_DJS] = {s["difference_kmeans_minus_real"]:+.5f} (K-means minus real; negative favors K-means)\n'
              f'[KM_P] = {s["exact_two_sided_signflip_p_raw"]:.5f} (raw, two-sided)\n[KM_FINDING] = {finding}\n'
              f'Complete pairs n={s["n"]}/13; invalid jobs={invalid}; 95% descriptive CI=[{ci[0]:+.5f}, {ci[1]:+.5f}]\n'
              'Interpret effect size, direction, per-question failures and uncertainty together. A nonsignificant p does not establish equivalence or robustness.\n')
    (out/'PLACEHOLDERS.txt').write_text(text);print(text)

if __name__=='__main__':main()
