"""Independent finite-difference check of KR stationarity from ln(det K).
The determinant-ratio identity avoids cancellation at large imaginary energy.
This checks the functional derivative, not just residual self-consistency.
"""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parent;KR=ROOT.parent/'finite_u_kr'
sys.path.insert(0,str(KR/'kr_package/src'))
import matrix_saddle as m

def variation(x0,x,U,eps,vo,s):
    p0,R0,n0,_,L0=m.unpack(x0);p,R,n,_,L=m.unpack(x)
    delta=s.delta(vo)
    G0=m.inv2(s.omega[:,None,None]*m.I-L0-R0@delta@R0)
    dK=L0-L+R0@delta@R0-R@delta@R
    Q=G0@dK
    t=np.trace(Q,axis1=1,axis2=2)+np.linalg.det(Q)
    # Real log magnitude evaluated without forming 1+t; complex log1p
    # can lose the O(y^-2) real part when t is mostly imaginary.
    logratio=.5*np.log1p(2*t.real+t.real*t.real+t.imag*t.imag)
    dOmega=np.trace(L-L0)/2-np.dot(s.w,logratio)
    mass=s.p.impurity_hybridization**2*s.p.cutoff**2/(4*np.pi)
    c=lambda R,L: mass*np.trace(R@R)+.5*np.trace((L-s.mu*m.I)@(L-s.mu*m.I))
    dOmega-=(c(R,L)-c(R0,L0))/(np.pi*s.ymax)
    return float(dOmega+np.trace((eps*m.I-L)@n-(eps*m.I-L0)@n0)+U*(p[3]-p0[3]))

out={'points':[]}
for name,file,idx in [('n1','kr_package/results/ep_n1_U0025.json',None),('low_fill_U02','kr_package/results/ep_mu000.json',-1),('occupied_U02','kr_package/results/occupied_gate_path.json',-1)]:
    a=json.loads((KR/file).read_text());r=a if idx is None else a[idx]
    s=m.Saddle(r['mu'],64,p=m.replace(m.P,impurity_hybridization=r['impurity_hybridization']))
    x0=np.array(r['x']);records=[]
    for displacement in (0.,.02):
        x=x0+displacement*np.array([.2,-.3,.4,.2,.01,-.01,.01])
        f=s.residual(x,r['U'],r['epsilon'],r['auxiliary_vertex']);p=m.unpack(x)[0];pv=p[1:];J=np.diag(pv)-np.outer(pv,pv)
        analytic=np.r_[J@(f[:3]*.02),f[3]*.02*(p[1]-p[2])/2,2*f[4:]]
        for h in (2e-5,1e-5,5e-6):
            steps=np.r_[np.repeat(h,4),np.repeat(h*.01,3)]
            num=[]
            for j,hj in enumerate(steps):
                shift=np.eye(7)[j]*hj
                num.append((variation(x,x+shift,r['U'],r['epsilon'],r['auxiliary_vertex'],s)-variation(x,x-shift,r['U'],r['epsilon'],r['auxiliary_vertex'],s))/(2*hj))
            error=float(np.max(np.abs(np.array(num)-analytic)))
            records.append({'displacement':displacement,'step':h,'functional_gradient_max':float(np.max(np.abs(num))),'derivative_discrepancy':error})
    out['points'].append({'name':name,'records':records})
    print(name,records,flush=True)
out['passed']=all(z['derivative_discrepancy']<2e-7 for p in out['points'] for z in p['records'])
(ROOT/'results/variational_functional.json').write_text(json.dumps(out,indent=2))
print('PASS',out['passed']);raise SystemExit(0 if out['passed'] else 1)
