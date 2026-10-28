"""Read-only audit of supplied 18 October saddle results. Outputs stay here."""
from pathlib import Path
import sys, json
import numpy as np
from scipy.optimize import root, minimize_scalar
from scipy.optimize._numdiff import approx_derivative

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / 'finite_u_kr'
sys.path.insert(0, str(SRC / 'scripts'))
sys.path.insert(0, str(SRC / 'kr_package/src'))
import matrix_saddle as m
import kr_certificate as c

out = {}
def save():
    (HERE / 'new_results_recheck.json').write_text(json.dumps(out, indent=2))
def brief(label, value):
    print(label, json.dumps(value), flush=True)

# Re-solve the strong-U, low-filling EP with a denser imaginary-axis grid.
old = json.loads((SRC/'results/cert_mu0_VO_U0.2.json').read_text())['ep']
s = m.Saddle(old['mu'], 48)
r = m.joint_ep(s, old['U'], np.array(old['y']), None, 'V_O')
out['low_filling_ep'] = r
brief('Strong-U EP', {k:r[k] for k in ['success','U','z','n_d','double_occupancy','joint_residual_max']})
summary, records = c.certificate(r, 'V_O', 1e-4, 1e-2, samples=128, order=48)
out['low_filling_certificate'] = summary
out['low_filling_records'] = records
brief('Strong-U loop', summary)
save()

# Follow the n=1 constrained curve using V_O as parameter, U as unknown.
# Refine its local maximum, instead of equating solver failure with absence.
rows = json.loads((SRC/'results/ep_n1_extended_VO.json').read_text())
seed = [r for r in rows if r.get('success',True) and not r.get('failed_attempt')][-1]
sf = m.Saddle(seed['mu'],48)
v0 = np.r_[seed['x'], np.array(seed['z'])/.01, seed['epsilon']/.01,
           seed['U']/.001, seed['mu']/.01]
def decode(v):
    return v[:7], complex(v[7],v[8])*.01, v[9]*.01, v[10]*.001, v[11]*.01
def constrained(v,vo):
    x,z,eps,U,mu = decode(v)
    sf.set_mu(mu)
    f=m.determinant(z,x,vo,sf.p); fz=m.ddet(z,x,vo,sf.p)
    return np.r_[sf.residual(x,U,eps,vo),f.real/1e-5,f.imag/1e-5,
                 fz.real/.01,fz.imag/.01,np.trace(m.unpack(x)[2])-1]
cache={}
def solve_par(vo):
    if vo in cache:return cache[vo]
    fun=lambda v:constrained(v,vo)
    sol=root(fun,v0,jac=lambda v:approx_derivative(fun,v,method='3-point',abs_step=1e-6),tol=1e-11)
    res=float(np.max(np.abs(fun(sol.x))))
    if res>1e-9:raise RuntimeError((vo,res))
    cache[vo]=(sol.x,res)
    return cache[vo]
opt=minimize_scalar(lambda vo:-decode(solve_par(vo)[0])[3],bounds=(.855,.859),method='bounded',options={'xatol':1e-10})
vo=float(opt.x); v,res=solve_par(vo)
x,z,eps,U,mu=decode(v)
sf.set_mu(mu)
row=sf.describe(x,U,eps,vo)
row.update(z=[z.real,z.imag],joint_residual=res)
h=2e-5
um=decode(solve_par(vo-h)[0])[3];up=decode(solve_par(vo+h)[0])[3]
row['dU_dVO']=(up-um)/(2*h)
row['d2U_dVO2']=(up-2*U+um)/h**2
row['constrained_parameter_jacobian_singular_values']=np.linalg.svd(approx_derivative(lambda w:constrained(w,vo),v,method='3-point',abs_step=1e-6),compute_uv=False).tolist()
out['refined_n1_constrained_fold']=row
brief('Refined n=1 fold',{k:row[k] for k in ['U','epsilon','auxiliary_vertex','mu','n_d','double_occupancy','joint_residual','dU_dVO','d2U_dVO2']})

# At this same chemical potential, release only the enforced n=1 condition.
# Test whether nearby EPs continue above the constrained maximum.
sf.set_mu(mu)
y=np.r_[x,z.real/.01,z.imag/.01,eps/.01,vo]
out['fixed_mu_beyond_constrained_fold']=[]
for unew in [U,.0032,.0035,.004]:
    rr=m.joint_ep(sf,unew,y,None,'V_O')
    out['fixed_mu_beyond_constrained_fold'].append(rr)
    brief('Fixed-mu EP',{k:rr[k] for k in ['U','success','z','n_d','double_occupancy','epsilon','auxiliary_vertex','joint_residual_max']})
    save()
    if not rr['success']:break
    y=np.array(rr['y'])
save()
