#!/usr/bin/env python3
"""Fixed-control interacting comparisons on optimized and Wilson baths."""
import argparse,json,time
from pathlib import Path
import numpy as np
import optimized_bath as b

POINTS=[(-.004,.003),(0.,.003),(.004,.003),(0.,.01),(0.,.1)]
def evaluate(H,U,eps,width):
    start=time.monotonic()
    state=b.cs.solve(H,U,50000,1e-11)
    print('ground solved',len(H)-2,'bath orbitals',time.monotonic()-start,flush=True)
    obs=b.cs.observables(state)
    rows=[]
    for re,im in POINTS:
        z=complex(re,im)
        G,Q,S=b.cs.response(state,z)
        sigma=b.cs.sigma_u(state,z)
        delta=b.physical([z+b.MU])[0]*(width/b.W)**2
        embedded=np.linalg.inv((z+b.MU-eps)*b.I-delta-sigma)
        rate=1j*(delta+sigma-(delta+sigma).conj().T)
        A=-(embedded-embedded.conj().T)/(2j*np.pi)
        rows.append({'z_relative_mu':[re,im],'sigma':b.cs.encode_matrix(sigma),'embedded_G':b.cs.encode_matrix(embedded),
                     'sum_rule_error':float(np.linalg.norm(S-b.I)),
                     'min_total_rate':float(np.linalg.eigvalsh(rate)[0]),'min_spectral_eigenvalue':float(np.linalg.eigvalsh(A)[0])})
    return {'observables':obs,'responses':rows,'seconds':time.monotonic()-start,
            'state_scope':'exact finite-bath Hermitian grand-canonical ground ensemble',
            'response_scope':'finite-bath interaction self-energy in analytic physical-sheet bath; no outgoing pole claim'}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--family',default='equilibrium');ap.add_argument('--sizes',default='8,10,12,14')
    ap.add_argument('--U',type=float,default=.2);ap.add_argument('--shift-invert',action='store_true');a=ap.parse_args()
    if a.shift_invert:
        import shift_invert_ground
        shift_invert_ground.install(b.cs)
    eps=b.EPS-a.U/2;width=.1
    out={'U':a.U,'mu':b.MU,'epsilon_d':eps,'W':width,'family':a.family,'shift_invert':a.shift_invert,'cases':[]}
    suffix='_shiftinvert' if a.shift_invert else ''
    path=b.ROOT/'results'/f'interacting_{a.family}{suffix}_U{a.U:g}.json'
    if a.family=='wilson':
        cfg=json.loads((b.SOURCE/'configs/submitted_baseline.json').read_text())
        cfg['current_mu']=b.MU;cfg['mapping']['z_shifts']=[.5]
        chains=b.fc.make_chains(cfg,b.MU,b.ROOT/'wilson_cache')
    else:
        fits=json.loads((b.ROOT/'results'/f'fits_{a.family}.json').read_text())['fits']
    for size in map(int,a.sizes.split(',')):
        row={'size_parameter':size,'size_means':'Wilson blocks (2 modes each)' if a.family=='wilson' else 'spinor bath orbitals (one fermion each)'}
        try:
            if a.family=='wilson':
                on,hop,v0,_=chains[0]
                H=b.fc.exact.build_one_body(on,hop,v0,eps-b.MU,width,b.W,size)
            else:
                fit=next(f for f in fits if f['bath_orbitals']==size)
                e,v=np.asarray(fit['energies']),np.asarray(fit['couplings'])
                H=b.one_body(e,v,eps,width)
            row['bath_orbitals']=len(H)-2
            row.update(evaluate(H,a.U,eps,width))
            obs=row['observables']
            print(a.family,size,'U',a.U,'nd',obs['nd'],'D',obs['double_occupancy'],'Q',obs['ground_charges'],'seconds',row['seconds'],flush=True)
        except Exception as exc:
            row['error']=type(exc).__name__+': '+str(exc);print(row['error'],flush=True)
        out['cases'].append(row);b.save(path,out)
if __name__=='__main__':main()
