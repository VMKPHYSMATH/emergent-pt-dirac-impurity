"""Independent scalar Coleman checks using the prior full stationary bath."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'finite_u_kr/kr_package/src'))
import matrix_saddle as m
import numpy as np
from scipy.optimize import root
from scipy.optimize._numdiff import approx_derivative
from scipy.special import polygamma

def effective_equilibrium(e,W,vo,mu=0,order=48):
    s=m.Saddle(mu,order);s.set_W(W)
    n,B=s.qp(m.I,e*m.I,vo)
    nf=np.trace(n).real;q=1-nf;force=np.trace(B).real/2
    return dict(nf=float(nf),q=float(q),r=float(np.sqrt(q)),delta_c=float(-force/q),epsilon_bare=float(e+force/q),W_bare=float(W/np.sqrt(q)),density=n.tolist())

def mapping(x):
    a=effective_equilibrium(x[0],x[1],m.VO0)
    return np.array([a['epsilon_bare'],a['W_bare']])

def solve_fixed_W():
    s=m.Saddle(0,48)
    def residual(y):
        z=.01*(y[0]+1j*y[1]);e=.01*y[2];vo=y[3];q=y[4]
        if q<=0:return np.ones(5)*1e4
        n,B=s.qp(np.sqrt(q)*m.I,e*m.I,vo)
        F=lambda zz:np.linalg.det(zz*m.I-e*m.I-q*m.bath.resonance_self_energy(zz,vo,m.P))
        f=F(z);df=m.bath.derivative(F,z,2e-6)
        return np.array([f.real/1e-5,f.imag/1e-5,df.real/.01,df.imag/.01,q+np.trace(n).real-1])
    sol=root(residual,[m.ZEP0.real/.01,m.ZEP0.imag/.01,m.EPS0/.01,m.VO0,.965],tol=1e-10)
    z=.01*(sol.x[0]+1j*sol.x[1]);e=.01*sol.x[2];vo=sol.x[3];q=sol.x[4]
    n,B=s.qp(np.sqrt(q)*m.I,e*m.I,vo)
    delta=-np.trace(B).real/(2*np.sqrt(q))
    return dict(z=[float(z.real),float(z.imag)],epsilon_tilde=float(e),epsilon_bare=float(e-delta),delta_c=float(delta),W_bare=.1,V_O=float(vo),q=float(q),r=float(np.sqrt(q)),nf=float(np.trace(n)),max_residual=float(max(abs(residual(sol.x)))))

def finite_temperature_map(T,N):
    e=m.EPS0;vo=m.VO0;p=m.P
    sn=np.zeros((2,2));force=0.
    for start in range(0,N,10000):
        j=np.arange(start,min(start+10000,N));z=1j*(2*j+1)*np.pi*T
        d=m.hybridization_array(z,vo,p)
        gf=m.inv2(z[:,None,None]*m.I-e*m.I-d)
        sn+=gf.real.sum(axis=0)
        force+=np.trace(d@gf,axis1=1,axis2=2).real.sum()
    tail=polygamma(1,N+.5)/(2*np.pi**2*T)
    mass=p.impurity_hybridization**2*p.cutoff**2/(4*np.pi)
    n=m.I/2+2*T*sn-e*m.I*tail
    force=2*T*force-2*mass*tail
    nf=np.trace(n);q=1-nf
    return dict(T=T,N=N,nf=float(nf),q=float(q),r=float(np.sqrt(q)),W_bare=float(.1/np.sqrt(q)),delta_c=float(-force/q),epsilon_bare=float(e+force/q))

def hf_slope(density=None,sigma_function=None):
    """F=F_z=0 response at U=0, fixed VO, tuning epsilon and W."""
    s=m.Saddle(0,64);n,_=s.qp(m.I,m.EPS0*m.I,m.VO0)
    if density is not None:n=density
    hf=np.trace(n)*m.I-n
    def eq(x,U):
        z=.01*(x[0]+1j*x[1]);e=.01*x[2];p=m.replace(m.P,impurity_hybridization=.1*x[3])
        F=lambda zz:np.linalg.det((zz-e)*m.I-m.bath.resonance_self_energy(zz,m.VO0,p)-U*(hf if sigma_function is None else sigma_function(zz)))
        f=F(z);fz=m.bath.derivative(F,z,2e-6)
        return np.array([f.real/1e-5,f.imag/1e-5,fz.real/.01,fz.imag/.01])
    x0=np.array([m.ZEP0.real/.01,m.ZEP0.imag/.01,m.EPS0/.01,1.])
    z0=root(lambda x:eq(x,0),x0,tol=1e-10).x
    J=approx_derivative(lambda x:eq(x,0),z0,method='3-point',abs_step=1e-6)
    b=approx_derivative(lambda u:eq(z0,u[0]),[0.],method='3-point',abs_step=1e-6).ravel()
    slope=-np.linalg.solve(J,b)
    return dict(dW_dU=float(.1*slope[3]),deps_dU=float(.01*slope[2]),dz_dU=(.01*slope[:2]).tolist(),density=n.tolist(),n_f=float(np.trace(n)))

def main():
    r={'fixed_effective_W':effective_equilibrium(m.EPS0,.1,m.VO0),'fixed_bare_W':solve_fixed_W()}
    J=approx_derivative(mapping,[m.EPS0,.1],method='3-point',abs_step=1e-6)
    r['control_mapping_J']=J.tolist();r['control_mapping_determinant']=float(np.linalg.det(J))
    r['temperature_mapping']=[finite_temperature_map(T,int(max(10000,np.ceil(1000/(2*np.pi*T))))) for T in (1e-3,.01,.1)]
    r['temperature_mapping_refined']=finite_temperature_map(.1,20000)
    r['Hartree_Fock_continuum_slope']=hf_slope()
    r['integration_orders']=[{'order':order,**effective_equilibrium(m.EPS0,.1,m.VO0,order=order)} for order in (24,48,96)]
    r['Matsubara_refinements']=[finite_temperature_map(T,N) for T in (1e-3,.01,.1) for N in (32000,64000,128000)]
    drift=max(abs(r['integration_orders'][-1][k]-r['integration_orders'][-2][k]) for k in ('nf','r','delta_c','W_bare','epsilon_bare'))
    td=[]
    for T in (1e-3,.01,.1):
        rows=[a for a in r['Matsubara_refinements'] if a['T']==T]
        td.append(max(abs(rows[-1][k]-rows[-2][k]) for k in ('nf','r','delta_c','W_bare','epsilon_bare')))
    r['order48_to96_max_change']=drift
    r['Matsubara64k_to128k_max_changes']=td
    r['passed']=bool(drift<1e-9 and max(td)<1e-8 and r['fixed_bare_W']['max_residual']<1e-9 and r['control_mapping_determinant']>0)
    (ROOT/'results/stationary_coleman.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))
    raise SystemExit(0 if r['passed'] else 1)

if __name__=='__main__':main()
