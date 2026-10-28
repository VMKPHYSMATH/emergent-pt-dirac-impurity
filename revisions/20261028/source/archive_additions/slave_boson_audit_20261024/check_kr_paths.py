"""Re-solve all stored accepted KR paths at higher quadrature orders.

Run from any directory. This validates the stated T=0 mean-field equations,
not the accuracy of KR against an exact interacting impurity solution.
"""
from pathlib import Path
import json, sys, time
import numpy as np
ROOT=Path(__file__).resolve().parent
KR=ROOT.parent/'finite_u_kr'
sys.path.insert(0,str(KR/'kr_package/src'))
import matrix_saddle as m
from validate import stability

def compact(r):
    keys=('U','epsilon','auxiliary_vertex','impurity_hybridization','mu','n_d','double_occupancy','Z_eigenvalues','R','lambda_matrix','z','residual_max','constraint_max','joint_residual_max','F','F_z','F_zz','singular_values','success','solver_success','order')
    return {k:r[k] for k in keys if k in r}

def metrics(a,b):
    keys=('epsilon','auxiliary_vertex','impurity_hybridization','mu','n_d','double_occupancy','R','Z_eigenvalues','lambda_matrix','z')
    return {k:float(np.max(np.abs(np.asarray(a[k])-np.asarray(b[k])))) for k in keys if k in a and k in b}

out={'scope':'Numerical convergence within the T=0 normal real-spin matrix KR/Gutzwiller saddle. No thermodynamic NRG claim.','paths':{},'tail_checks':[]}
paths=['kr_package/results/ep_mu000.json','kr_package/results/ep_n1.json','kr_package/results/ep_n1_W.json','kr_package/results/occupied_gate_path.json','results/ep_n1_extended_VO.json','results/ep_n1_extended_W.json']
start=time.time()
for name in paths:
    rows=json.loads((KR/name).read_text()); checked=[]; skipped=[]
    for i,r in enumerate(rows):
        if not r.get('success',False) or r.get('failed_attempt',False):
            skipped.append({'index':i,'U':r['U'],'reason':'stored failed attempt; not an accepted EP'});continue
        ep='y' in r
        refined=[]
        for order in (32,64):
            s=m.Saddle(r['mu'],order,p=m.replace(m.P,impurity_hybridization=r['impurity_hybridization']))
            if ep: q=m.joint_ep(s,r['U'],np.array(r['y']),r.get('target_filling',1.0 if len(r['y'])==12 else None),r.get('retuned_control','W' if name.endswith('_W.json') else 'V_O'))
            else: q=s.solve(r['U'],r['epsilon'],r['auxiliary_vertex'],np.array(r['x']))
            refined.append(q)
        q=refined[-1]
        # Held-state independent finer integration, no new optimization.
        fine=m.Saddle(q['mu'],96,p=m.replace(m.P,impurity_hybridization=q['impurity_hybridization']))
        fcheck=fine.describe(np.array(q['x']),q['U'],q['epsilon'],q['auxiliary_vertex'])
        dif=metrics(*refined)
        ok=all(x['success'] for x in refined) and fcheck['residual_max']<1e-8 and max(dif.values())<1e-7
        checked.append({'index':i,'U':r['U'],'passed':bool(ok),'order32_to64_absolute_changes':dif,'order96_held_state_residual':fcheck['residual_max'],'order96_constraint':fcheck['constraint_max'],'refined':compact(q)})
        print(name,i,'U',r['U'],'PASS' if ok else 'FAIL','res96',fcheck['residual_max'],flush=True)
    out['paths'][name]={'checked':checked,'skipped':skipped,'passed':all(x['passed'] for x in checked)}
    (ROOT/'results/kr_paths.json').write_text(json.dumps(out,indent=2))

# Tail cutoff, unrelated to increasing the per-decade Gauss order.
from numpy.polynomial.legendre import leggauss
for label, filename, idx in [('n1_ep','kr_package/results/ep_n1_U0025.json',None),('low_filling_U02','kr_package/results/ep_mu000.json',-1),('occupied_U02','kr_package/results/occupied_gate_path.json',-1)]:
    rows=json.loads((KR/filename).read_text());r=rows if idx is None else rows[idx]
    data=[]
    for emax in (4,5,6):
        s=m.Saddle(r['mu'],64,p=m.replace(m.P,impurity_hybridization=r['impurity_hybridization']))
        g,w=leggauss(64); edges=np.r_[0.,10.**np.arange(-11,emax+1,dtype=float)]
        yy=[];ww=[]
        for lo,hi in zip(edges[:-1],edges[1:]):
            yy.extend((lo+hi)/2+(hi-lo)*g/2);ww.extend((hi-lo)*w/(2*np.pi))
        s.y=np.array(yy);s.w=np.array(ww);s.ymax=edges[-1];s.omega=s.mu+1j*s.y;s._vo=None
        nq,B=s.qp(np.array(r['R']),np.array(r['lambda_matrix']),r['auxiliary_vertex'])
        data.append({'cutoff':s.ymax,'density':nq.tolist(),'force':B.tolist()})
    delta_n=float(np.max(np.abs(np.array(data[1]['density'])-data[2]['density'])))
    delta_B=float(np.max(np.abs(np.array(data[1]['force'])-data[2]['force'])))
    out['tail_checks'].append({'point':label,'values':data,'density_change_1e5_to1e6':delta_n,'force_change_1e5_to1e6':delta_B,'passed':delta_n<1e-9 and delta_B<1e-9})

out['passed']=all(v['passed'] for v in out['paths'].values()) and all(x['passed'] for x in out['tail_checks'])
out['elapsed_seconds']=time.time()-start
(ROOT/'results/kr_paths.json').write_text(json.dumps(out,indent=2))
print('OVERALL',out['passed'],'seconds',out['elapsed_seconds'],flush=True)
raise SystemExit(0 if out['passed'] else 1)
