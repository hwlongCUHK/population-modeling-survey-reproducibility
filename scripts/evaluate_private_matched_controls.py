"""Offline evaluation: target counts are first introduced after inference."""
import argparse,itertools,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=None
ORIG6=[q['id'] for q in sorted(json.loads((ROOT/'results/question_index.json').read_text()),key=lambda q:q['original_six_rank'] or 99) if q['original_six']]

def js(p,q):
    p=np.asarray(p,dtype=float); q=np.asarray(q,dtype=float)
    assert p.ndim==q.ndim==1 and p.shape==q.shape
    assert np.isfinite(p).all() and np.isfinite(q).all() and (p>=0).all() and (q>=0).all()
    p=p/p.sum();q=q/q.sum();m=(p+q)/2
    return float(sum(np.sum(x[x>0]*np.log(x[x>0]/m[x>0])) for x in [p,q])/2)

def paired(a,b,keys):
    common=[k for k in keys if a.get(k) is not None and b.get(k) is not None]
    if not common:return dict(n=0)
    d=np.array([a[k]-b[k] for k in common])
    stats=np.array([np.mean(d*np.array(s)) for s in itertools.product([-1,1],repeat=len(d))])
    p=float(np.mean(np.abs(stats)>=abs(d.mean())-1e-14))
    rng=np.random.default_rng(20261026)
    boot=rng.choice(d,size=(20000,len(d)),replace=True).mean(axis=1)
    return dict(n=len(d),keys=common,mean_a=float(np.mean([a[k] for k in common])),mean_b=float(np.mean([b[k] for k in common])),
                difference_a_minus_b=float(d.mean()),ci95_question_bootstrap=np.quantile(boot,[.025,.975]).tolist(),
                exact_two_sided_signflip_p=p,a_better=int((d<0).sum()),b_better=int((d>0).sum()),
                per_question_differences=dict(zip(common,d.tolist())))

def main():
    global RUN
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=Path,required=True)
    args=ap.parse_args();RUN=args.run_dir.resolve()
    if RUN.is_relative_to(ROOT):raise SystemExit('Use a private run directory outside the repository')
    target=json.loads((RUN/'targets.json').read_text()); keys=list(target)
    ssr=json.loads((RUN/'ssr_predictions.json').read_text())
    llm=json.loads((RUN/'llm_predictions.json').read_text()) if (RUN/'llm_predictions.json').exists() else {}
    scores={}; perq={}
    for k in keys:
        perq[k]={}
        for arm,v in ssr[k]['arms'].items():
            metric=js(target[k],v['prediction']);name='SSR/'+arm
            scores.setdefault(name,{})[k]=metric;perq[k][name]=metric
        for arm,v in llm.get(k,{}).items():
            metric=js(target[k],v['prediction']) if v['prediction'] is not None else None;name='LLM/'+arm
            scores.setdefault(name,{})[k]=metric;perq[k][name]=metric
        r=[perq[k].get(f'LLM/random_{s}') for s in [101,202,303,404,505]]
        if llm:
            metric=float(np.mean(r)) if all(v is not None for v in r) else None
            scores.setdefault('LLM/random_mean_JS',{})[k]=metric;perq[k]['LLM/random_mean_JS']=metric
    comps=[('SSR/direct_clustered','SSR/cluster_selected'),('SSR/direct_full','SSR/cluster_selected'),
           ('SSR/cluster_selected','SSR/random_clustered'),('LLM/real','LLM/random_mean_JS'),
           ('LLM/direct_clustered','LLM/real'),('LLM/direct_full','LLM/real'),
           ('LLM/direct_clustered','LLM/random_101'),('LLM/direct_full','LLM/random_101')]
    comps += [('LLM/real',f'LLM/random_{s}') for s in [101,202,303,404,505]]
    comparisons={}
    for subset,kk in [('all13',keys),('original6',ORIG6)]:
        comparisons[subset]={f'{a} vs {b}':paired(scores[a],scores[b],kk) for a,b in comps if a in scores and b in scores}
    summary={}
    for subset,kk in [('all13',keys),('original6',ORIG6)]:
        summary[subset]={}
        for arm,v in scores.items():
            vals=[v[k] for k in kk if v.get(k) is not None]
            summary[subset][arm]=dict(n=len(vals),mean_js=float(np.mean(vals)) if vals else None)
    # Three prespecified main tests; show raw and Holm-adjusted p values.
    primary=['SSR/direct_clustered vs SSR/cluster_selected','SSR/direct_full vs SSR/cluster_selected','LLM/real vs LLM/random_mean_JS']
    vals=[(k,comparisons['all13'][k]) for k in primary if k in comparisons['all13'] and comparisons['all13'][k]['n']]
    running=0
    for i,(name,v) in enumerate(sorted(vals,key=lambda kv:kv[1]['exact_two_sided_signflip_p'])):
        running=max(running,min(1,v['exact_two_sided_signflip_p']*(len(vals)-i)))
        v['holm_p_primary_family']=running
    inv=max(v['grouping_invariance_max_abs'] for v in ssr.values())
    report=dict(summary=summary,comparisons=comparisons,per_question=perq,
                ssr_grouping_invariance_max_abs=inv,
                notes=['New matched protocol; do not combine with historical paper point estimates.',
                       'CIs resample questions within one domain and are descriptive.',
                       'Random grouping repetitions share posts and are not independent question samples.',
                       'No fallback distributions for invalid LLM responses.'])
    (RUN/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    lines=['# Rebuttal control results','', 'JS uses natural logarithms; lower is better. New fixed-anchor / greedy-chat protocol.', '',
           '| Estimator / arm | All 13 JS | n | Original 6 JS | n |','|---|---:|---:|---:|---:|']
    for arm in scores:
        a=summary['all13'][arm];b=summary['original6'][arm]
        fmt=lambda x:'missing' if x is None else f'{x:.6f}'
        lines.append(f'| {arm} | {fmt(a["mean_js"])} | {a["n"]} | {fmt(b["mean_js"])} | {b["n"]} |')
    lines+=['','## Paired comparisons (all 13)','', '| A vs B | n | JS(A)-JS(B) | 95% descriptive CI | exact p | Holm p (primary) |','|---|---:|---:|---|---:|---:|']
    for name,v in comparisons['all13'].items():
        if not v['n']:continue
        ci=v['ci95_question_bootstrap'];hp=v.get('holm_p_primary_family')
        lines.append(f'| {name} | {v["n"]} | {v["difference_a_minus_b"]:.6f} | [{ci[0]:.6f}, {ci[1]:.6f}] | {v["exact_two_sided_signflip_p"]:.5f} | {hp if hp is not None else "descriptive"} |')
    lines+=['',f'Fixed-post SSR grouping invariance maximum absolute error: {inv:.3g}.', '',
            'Human labels are pending; annotation forms are not completed evidence validation.','',
            'The cached cluster selection is shared across questions. These controls do not validate clusters as demographic population strata.']
    (RUN/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
