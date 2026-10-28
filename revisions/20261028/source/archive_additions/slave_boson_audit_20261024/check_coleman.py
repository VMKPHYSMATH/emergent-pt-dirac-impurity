"""Independent finite-cutoff, scalar Coleman saddle and grid-convergence audit.

The Markov kernel is a separate supporting benchmark, not the full driven bath.
Finite-T digamma adds the finite lower-edge correction -log(1+z/D), with
exponentially small thermal edge error for the tested D/T >= 2e6. Adaptive
real-axis quadrature checks that correction independently.
"""
from pathlib import Path
from dataclasses import replace
import json, math, sys, time
import numpy as np
from scipy.optimize import root
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'finite_u_kr/scripts'))
import sb_kondo_closed_form as cf
import saddle_gate_copy as gate
import saddle_gate_legacy as legacy

def primitive(z,D,T):
    return cf.L_T0(z,D) if T==0 else cf.L_T(z,D,T)-np.log1p(z/D)

def eq(r2,et,D0,Gp,T,use_quad=False):
    s0=np.sqrt(complex(D0*D0-Gp*Gp));zs=[et-1j*r2*.12+r2*s0,et-1j*r2*.12-r2*s0]
    L=[cf.L_quad(z,4.,T) if use_quad and T>0 else primitive(z,4.,T) for z in zs]
    nf=-sum(L).imag/np.pi
    dc=-.12*sum(L).real/np.pi+(s0*(L[0]-L[1])).imag/np.pi
    return np.array([r2+nf-1, dc-(et+1)]),zs

def solve(D0,Gp,T,seed=None):
    r20=cf.closed_r2(-1,.12,D0,Gp,4);scale=r20*.12
    f=lambda x:eq(np.exp(x[0]),x[1]*scale,D0,Gp,T)[0]
    x0=[np.log(r20),0.] if seed is None else [np.log(seed['r2']),seed['et']/scale]
    sol=root(f,x0,tol=1e-10);res=float(np.max(np.abs(f(sol.x))))
    r2=float(np.exp(sol.x[0]));et=float(sol.x[1]*scale);_,zs=eq(r2,et,D0,Gp,T)
    if res>1e-10:raise RuntimeError(f'Saddle residual {res}')
    return {'r':np.sqrt(r2),'r2':r2,'et':et,'delta_c':1+et,'constraint_stationarity_residual':res,'solver_success':bool(sol.success),'temperature':T,'D0':D0,'GammaPT':Gp,'z_plus':[zs[0].real,zs[0].imag],'z_minus':[zs[1].real,zs[1].imag]}

start=time.time();out={'scope':'U=infinity, Q=1 spin-scalar Coleman saddle of the passive constant-core Markov benchmark. Not the finite-U KR calculation.','reference':[],'grid':[],'old_grid_diagnostic':{},'temperature':[],'production_solver':[]}
for convention in ('archive','fixed_Delta'):
    for beta in (.35,.40,.45,.50,.55):
        D0=.075+.05*beta if convention=='archive' else .10
        for T in (0.,1e-6):
            r=solve(D0,.2*beta,T);r.update(beta=beta,convention=convention)
            if T:
                qr,_=eq(r['r2'],r['et'],D0,.2*beta,T,True)
                r['adaptive_quadrature_residual']=float(np.max(np.abs(qr)))
            out['reference'].append(r)
for beta in (.35,.50,.55):
    ref=next(r for r in out['reference'] if r['convention']=='archive' and r['beta']==beta and r['temperature']==1e-6)
    width=ref['r2']*.12
    # Root solve on each grid with scaled level variable, then compare to the
    # independently evaluated logarithm/digamma/adaptive integral.
    for refine in (1,2,4):
        def f(x):
            vals=gate.evaluate_saddle(float(np.exp(x[0]/2)),float(1+x[1]*width),beta,refine)
            return np.array([vals['constraint_residual'],vals['stationarity_divided_residual']])
        sol=root(f,[np.log(ref['r2']),ref['et']/width],tol=1e-9)
        r=float(np.exp(sol.x[0]/2));dc=float(1+sol.x[1]*width)
        vals=gate.evaluate_saddle(r,dc,beta,refine)
        independent,_=eq(r*r,dc-1,.075+.05*beta,.2*beta,1e-6)
        row={'beta':beta,'refinement':refine,'r':r,'delta_c':dc,'relative_r_error':abs(r/ref['r']-1),'on_grid_residual':float(np.max(np.abs(f(sol.x)))),'independent_residual':float(np.max(np.abs(independent))),'points':int(vals['omega_points']),'min_rate':vals['rate_min_eigenvalue'],'min_spectral':vals['spectral_min_eigenvalue'],'FDR_absolute':vals['fdr_abs_residual'],'charge_residual':vals['charge_continuity_abs_residual']}
        out['grid'].append(row);print('GRID',row,flush=True)
    # Production safeguarded Brent solver: explicit acceptance flag required.
    p=replace(gate.P,r_search_min=.004,r_search_max=.02,r_search_points=7)
    row=gate.solve_seed(beta,gate.SEEDS[0],p)
    out['production_solver'].append({k:row.get(k) for k in ('beta0','status','converged','r','delta_c','constraint_residual','stationarity_divided_residual','grid_refinement_error','failure_reason')})
    print('PRODUCTION',out['production_solver'][-1],flush=True)
oldr=.009807411733055882;olddc=1.0000001191596621
for key,mod in [('original',legacy),('corrected',gate)]:
    q=mod.evaluate_saddle(oldr,olddc,.5,2)
    out['old_grid_diagnostic'][key]={k:q[k] for k in ('constraint_residual','stationarity_divided_residual','omega_points')}
for T in (0.,1e-8,1e-7,3e-7,1e-6,2e-6):out['temperature'].append(solve(.1,.1,T))
end=[r for r in out['grid'] if r['refinement']==4]
out['passed']=bool(all(r.get('adaptive_quadrature_residual',0)<1e-10 for r in out['reference']) and all(r['relative_r_error']<2e-6 and r['independent_residual']<1e-6 and r['on_grid_residual']<1e-8 and r['min_rate']>0 and r['min_spectral']>=-1e-12 for r in end) and all(r['converged']==1 for r in out['production_solver']))
out['elapsed_seconds']=time.time()-start
(ROOT/'results/coleman.json').write_text(json.dumps(out,indent=2))
print('OVERALL',out['passed'],'seconds',out['elapsed_seconds'])
raise SystemExit(0 if out['passed'] else 1)
