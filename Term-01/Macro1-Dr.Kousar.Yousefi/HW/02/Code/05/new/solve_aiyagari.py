import numpy as np
from scipy.sparse import coo_matrix, eye
from scipy.sparse.linalg import spsolve
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

BETA, ALPHA, DELTA = .94, .36, .08
P = np.array([[.9,.1],[.1,.9]])
Z = np.array([.5,1.5])

def solve(n=600, amax=40.):
    a = amax*np.linspace(0,1,n)**2
    def transition(g):
        j=np.clip(np.searchsorted(a,g,side='right')-1,0,n-2)
        q=(g-a[j])/(a[j+1]-a[j])
        rows=[]; cols=[]; data=[]
        for z in range(2):
            for zz in range(2):
                rows.extend([z*n+np.arange(n)]*2)
                cols.extend([zz*n+j[z], zz*n+j[z]+1])
                data.extend([P[z,zz]*(1-q[z]),P[z,zz]*q[z]])
        return coo_matrix((np.concatenate(data),(np.concatenate(rows),np.concatenate(cols))),shape=(2*n,2*n)).tocsr()
    def household(K):
        r=ALPHA*K**(ALPHA-1)-DELTA
        w=(1-ALPHA)*K**ALPHA
        cash=(1+r)*a[None,:]+w*Z[:,None]
        V=np.log(cash)/(1-BETA)
        def improve(V):
            EV=P@V
            def obj(x):
                return np.log(cash-x)+BETA*np.array([np.interp(x[z],a,EV[z]) for z in range(2)])
            lo=np.zeros_like(cash); hi=np.minimum(cash-1e-12,amax)
            ratio=(np.sqrt(5)-1)/2
            x=hi-ratio*(hi-lo); y=lo+ratio*(hi-lo)
            fx=obj(x); fy=obj(y)
            for _ in range(65):
                left=fx>fy
                hi=np.where(left,y,hi); lo=np.where(left,lo,x)
                x=hi-ratio*(hi-lo); y=lo+ratio*(hi-lo)
                fx=obj(x); fy=obj(y)
            g=(lo+hi)/2
            # Include the borrowing-constraint boundary explicitly.
            f=obj(g); f0=obj(np.zeros_like(g))
            g=np.where(f0>=f,0.,g)
            return g,obj(g)
        for it in range(150):
            g,_=improve(V)
            Q=transition(g)
            u=np.log(cash-g)
            Vnew=spsolve(eye(2*n,format='csr')-BETA*Q,u.ravel()).reshape(2,n)
            diff=np.max(abs(Vnew-V)); V=Vnew
            if diff<1e-10: break
        _,TV=improve(V)
        bell=np.max(abs(TV-V))
        mu=np.full(2*n,1/(2*n))
        for dit in range(100000):
            new=Q.T@mu
            if np.max(abs(new-mu))<1e-13:
                mu=new; break
            mu=new
        stat=np.max(abs(Q.T@mu-mu))
        mu=mu.reshape(2,n)
        Ks=np.sum(mu*a)
        return dict(K=K,Ks=Ks,r=r,w=w,Y=K**ALPHA,C=float(np.sum(mu*(cash-g))),capital_error=Ks-K,bellman_error=float(bell),distribution_error=float(stat),mass_zero=float(mu[:,0].sum()),upper_mass=float(mu[:,-1].sum()),max_policy=float(g.max()),iterations=it+1,distribution_iterations=dit+1),g,V,mu
    def excess(K):
        d,*_=household(K)
        print(n,round(K,9),round(d['capital_error'],9),flush=True)
        return d['capital_error']
    root=brentq(excess,4.,9.,xtol=1e-10)
    d,g,V,mu=household(root)
    np.savez(f'aiyagari_{n}.npz',a=a,g=g,V=V,mu=mu)
    return d,a,g,V,mu

if __name__=='__main__':
    results=[]
    for n in [600,1200]:
        d,a,g,V,mu=solve(n)
        results.append(d|{'grid_size':n})
        print(json.dumps(results[-1],indent=2),flush=True)
    with open('aiyagari_results.json','w') as f: json.dump(results,f,indent=2)
    fig,ax=plt.subplots(1,2,figsize=(12,4.5))
    for z in range(2): ax[0].plot(a,g[z],label=f'z = {Z[z]}')
    ax[0].plot(a,a,'k--',lw=1,label="a' = a")
    ax[0].set(xlim=(0,20),ylim=(0,20),xlabel='Current assets a',ylabel="Next-period assets a'",title='Optimal saving policy')
    ax[0].legend(); ax[0].grid(alpha=.3)
    bins=np.linspace(0,20,51)
    for z in range(2):
        hist,_=np.histogram(a,bins=bins,weights=mu[z])
        ax[1].stairs(hist,bins,label=f'z = {Z[z]}')
    ax[1].set(xlabel='Assets a',ylabel='Joint probability mass per bin',title='Stationary wealth distribution')
    ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig('aiyagari_solution.png',dpi=180)
