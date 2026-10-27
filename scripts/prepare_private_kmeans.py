"""Fixed-evidence, capacity-constrained K-means; never loads survey outcomes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from population_survey.prompts import prompt

BASE=Path(__file__).resolve().parents[1]

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n'); tmp.replace(path)

def assignment(cost):
    """Minimum-cost square assignment, Hungarian primal-dual algorithm (O(n^3)).

    Returns one distinct column for each row. Implemented locally to keep offline
    grouping/evaluation dependent only on NumPy; brute-force checked in tests.
    """
    cost=np.asarray(cost,dtype=np.float64)
    if cost.ndim!=2 or cost.shape[0]!=cost.shape[1] or not np.isfinite(cost).all():
        raise ValueError('Expected a finite square cost matrix')
    n=len(cost); u=np.zeros(n+1); v=np.zeros(n+1)
    p=np.zeros(n+1,dtype=int); way=np.zeros(n+1,dtype=int)
    for i in range(1,n+1):
        p[0]=i; j0=0; minv=np.full(n+1,np.inf); used=np.zeros(n+1,dtype=bool)
        while True:
            used[j0]=True; i0=p[j0]; delta=np.inf; j1=0
            for j in range(1,n+1):
                if not used[j]:
                    cur=cost[i0-1,j-1]-u[i0]-v[j]
                    if cur<minv[j]: minv[j]=cur; way[j]=j0
                    if minv[j]<delta: delta=minv[j]; j1=j
            for j in range(n+1):
                if used[j]: u[p[j]]+=delta; v[j]-=delta
                else: minv[j]-=delta
            j0=j1
            if p[j0]==0: break
        while True:
            j1=way[j0]; p[j0]=p[j1]; j0=j1
            if j0==0: break
    result=np.empty(n,dtype=int)
    for j in range(1,n+1): result[p[j]-1]=j-1
    return result

def balanced_kmeans(x,k=10,capacity=8,seed=42,restarts=10,max_iter=100):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or len(x)!=k*capacity or not np.isfinite(x).all():
        raise ValueError('Invalid features / capacities')
    rng=np.random.default_rng(seed); best=None; diagnostics=[]
    for start in range(restarts):
        chosen=[int(rng.integers(len(x)))]; nearest=((x-x[chosen[0]])**2).sum(axis=1)
        while len(chosen)<k:
            probs=nearest.copy(); probs[chosen]=0
            nxt=int(rng.choice(len(x),p=probs/probs.sum())) if probs.sum()>1e-15 else next(i for i in range(len(x)) if i not in chosen)
            chosen.append(nxt); nearest=np.minimum(nearest,((x-x[nxt])**2).sum(axis=1))
        centers=x[chosen].copy(); previous=None; trace=[]; converged=False
        for _ in range(max_iter):
            dist=((x[:,None,:]-centers[None,:,:])**2).sum(axis=2)
            labels=assignment(np.repeat(dist,capacity,axis=1))//capacity
            centers=np.stack([x[labels==j].mean(axis=0) for j in range(k)])
            objective=float(((x-centers[labels])**2).sum()); trace.append(objective)
            if len(trace)>1 and trace[-1]>trace[-2]+1e-8: raise RuntimeError('Objective increased')
            if previous is not None and np.array_equal(previous,labels): converged=True; break
            previous=labels.copy()
        diagnostics.append({'restart':start,'sse':objective,'iterations':len(trace),'converged':converged,'trace':trace})
        if best is None or objective<best[0]: best=(objective,labels.copy(),start)
    if not diagnostics[best[2]]['converged']: raise RuntimeError('Best restart did not converge; inspect before proceeding')
    return best[1], {'seed':seed,'restarts':restarts,'best_restart':best[2], 'sse':best[0],'runs':diagnostics}

def build_jobs(data, groups):
    byid={p['id']:p['text'] for p in data['posts']}; ids=list(byid); jobs=[]
    for arm,gg in groups.items():
        if len(gg)!=10 or any(len(g)!=8 for g in gg) or sorted(sum(gg,[]))!=sorted(ids):
            raise ValueError('Every arm must use each of the same 80 posts exactly once, in 10 groups of 8')
    for q in data['questions']:
        for arm,gg in groups.items():
            for gi,g in enumerate(gg):
                jobs.append({'job_id':f'{q["key"]}|{arm}|{gi}','key':q['key'],'arm':arm,'group':gi,
                             'post_ids':g,'n_options':len(q['options']),'prompt':prompt(q,[byid[i][:100] for i in g])})
    return jobs

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--reference-jobs',type=Path,help='Optional private archived real jobs for exact prompt comparison')
    ap.add_argument('--out',type=Path,required=True)
    group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--encoder',type=Path,help='Existing LOCAL bge-small-zh-v1.5 model directory; no downloads')
    group.add_argument('--features-npz',type=Path,help='External features with arrays ids and vectors; allow_pickle=False')
    ap.add_argument('--feature-description',help='Required with external features: encoder, text window, normalization, reduction')
    ap.add_argument('--device',default='cpu')
    args=ap.parse_args()
    if args.out.resolve().is_relative_to(BASE): raise SystemExit('Use a private output directory outside the repository')
    if args.out.exists() and any(args.out.iterdir()): raise SystemExit('Use an empty output directory to avoid overwriting a run')
    data=json.loads(args.input.read_text()); ids=[p['id'] for p in data['posts']]
    if args.features_npz:
        if not args.feature_description: raise SystemExit('--feature-description is required')
        with np.load(args.features_npz,allow_pickle=False) as z:
            if z['ids'].tolist()!=ids: raise SystemExit('Feature IDs/order do not exactly match data/posts')
            x=z['vectors'].astype(np.float64)
        provenance={'mode':'external_features','description':args.feature_description,'features_sha256':digest(args.features_npz)}
    else:
        if not args.encoder.is_dir(): raise SystemExit('Encoder must be an existing local model directory')
        os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
        from sentence_transformers import SentenceTransformer
        import sentence_transformers
        model=SentenceTransformer(str(args.encoder.resolve()),device=args.device,local_files_only=True)
        x=model.encode([p['text'] for p in data['posts']],batch_size=32,normalize_embeddings=True,show_progress_bar=True)
        provenance={'mode':'fresh_embeddings','encoder':str(args.encoder.resolve()),
                    'representation':'Full post text, SentenceTransformer default truncation, L2-normalized embeddings; no UMAP',
                    'max_seq_length':model.max_seq_length,'sentence_transformers_version':sentence_transformers.__version__}
    labels,stats=balanced_kmeans(x)
    # Canonical ordering relative to the archived 80-post list; real-arm order remains byte-identical.
    km=[[ids[i] for i in range(80) if labels[i]==j] for j in range(10)]
    km.sort(key=lambda g:ids.index(g[0]))
    groups={'real':data['real_groups'],'kmeans_balanced':km}; jobs=build_jobs(data,groups)
    if args.reference_jobs:
        archived=json.loads(args.reference_jobs.read_text())
        if [j for j in jobs if j['arm']=='real']!=archived: raise RuntimeError('Real prompts differ from the provided private reference')
    args.out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.out/'features.npz',ids=np.asarray(ids),vectors=x)
    save(args.out/'groups.json',groups); save(args.out/'jobs.json',jobs)
    save(args.out/'grouping_provenance.json',{'input_sha256':digest(args.input),'features':provenance,
         'algorithm':'capacity-constrained Lloyd iterations with optimal slot assignment; Euclidean SSE',
         'fixed_settings':{'k':10,'capacity':8,'seed':42,'restarts':10,'max_iter':100},
         'diagnostics':stats,'target_distributions_used':False,'jobs':len(jobs),
         'feature_file_sha256':digest(args.out/'features.npz'),'jobs_sha256':digest(args.out/'jobs.json')})
    print(f'Prepared {len(jobs)} jobs (130 real + 130 capacity-constrained K-means); no model inference run.')

if __name__=='__main__': main()
