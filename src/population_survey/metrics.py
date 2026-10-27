"""Natural-log JS divergence and paired question-level statistics."""
import itertools
import numpy as np

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
