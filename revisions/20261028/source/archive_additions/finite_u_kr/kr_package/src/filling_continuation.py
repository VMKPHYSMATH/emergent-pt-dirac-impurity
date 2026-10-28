import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
import matrix_saddle as m

ap=argparse.ArgumentParser();ap.add_argument('--filling',type=float,default=1.);ap.add_argument('--order',type=int,default=24);ap.add_argument('--out',default='results/ep_n1.json');ap.add_argument('--umax',type=float,default=.2);ap.add_argument('--control',choices=['V_O','W'],default='V_O')
a=ap.parse_args();s=m.Saddle(order=a.order)
def filling(mu):
    s.set_mu(mu)
    return np.trace(s.qp(m.I,m.EPS0*m.I,m.VO0)[0])-a.filling
mu=brentq(filling,-.05,.05,xtol=1e-14);s.set_mu(mu)
x=s.initial();y=np.r_[x,m.ZEP0.real/.01,m.ZEP0.imag/.01,m.EPS0/.01,m.VO0,mu/.01]
if a.control=='W':y[10]=1.
rows=[];U=0;step=.0001
while U<=a.umax+1e-12:
    row=m.joint_ep(s,U,y,a.filling,a.control)
    print(json.dumps({k:row[k] for k in ['U','success','mu','epsilon','auxiliary_vertex','impurity_hybridization','z','double_occupancy','Z_eigenvalues','density_eigenvalues','joint_residual_max']}),flush=True)
    if row['success']:
        rows.append(row);y=np.array(row['y'])
        step=min(step*1.45,.005)
        if U>=a.umax:break
        U=min(U+step,a.umax)
    else:
        if step<1e-7:
            row['failed_attempt']=True;rows.append(row);break
        step/=2
        U=rows[-1]['U']+step
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(rows,indent=2))
Path(a.out).write_text(json.dumps(rows,indent=2))
