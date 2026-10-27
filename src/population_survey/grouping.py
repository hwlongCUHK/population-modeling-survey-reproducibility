"""Capacity-constrained K-means, using NumPy only."""
import numpy as np

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
