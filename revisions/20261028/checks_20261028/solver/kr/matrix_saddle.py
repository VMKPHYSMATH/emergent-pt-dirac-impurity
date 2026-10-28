"""Zero-temperature, normal, real-spin finite-U KR/Gutzwiller saddle.

The variational local probabilities are (e,p_plus,p_minus,d), in a freely
rotating natural spin basis.  The spin constraint matrix is unrestricted
within the real symmetric sector of this real microscopic Hamiltonian.
No scalar-Z or unpolarized constraint is imposed. This is a mean-field
approximation, not an interacting NRG calculation or an exact many-body EP.

Functional: L=Omega_qp(R,Lambda)+(epsilon I-Lambda):n+U*d.
Stationarity is evaluated with the complete physical-sheet continuum bath.
Resonances subsequently use the outgoing bath continuation. T=0 throughout.
"""
from __future__ import annotations
import json
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import root
from scipy.optimize._numdiff import approx_derivative
from scipy.special import softmax
import full_k_two_cut_ep_certificate as bath

I=np.eye(2); X=np.array([[0.,1.],[1.,0.]]); Z=np.diag([1.,-1.])
PAULI=np.array([I,X,Z])
P=replace(bath.Parameters(),soc=.05)
EPS0=.0151703497834
VO0=1.21342200961
ZEP0=.011972834346-0.002807216274j

def components(a):
    return np.einsum('aij,ji->a',PAULI,a)/2

def inv2(a):
    det=a[...,0,0]*a[...,1,1]-a[...,0,1]*a[...,1,0]
    out=np.empty_like(a)
    out[...,0,0]=a[...,1,1];out[...,1,1]=a[...,0,0]
    out[...,0,1]=-a[...,0,1];out[...,1,0]=-a[...,1,0]
    return out/det[...,None,None]

@lru_cache(None)
def integration_grid(order=20):
    # Gaussian integration on decades; analytic high-frequency tails below.
    edges=np.r_[0.,10.**np.arange(-11,6,dtype=float)]
    x,w=leggauss(order)
    yy=[];ww=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        yy.extend((hi+lo)/2+(hi-lo)*x/2)
        ww.extend(w*(hi-lo)/2/np.pi)
    return np.array(yy),np.array(ww),edges[-1]

def hybridization_array(z,vo,p=P):
    z=np.asarray(z,complex)
    ev,odd=bath.auxiliary_terms(z,vo,p)
    q=z-ev
    x1,x2=bath.radial_roots(q,odd,p)
    l1=np.log((p.cutoff**2-x1)/(-x1))
    l2=np.log((p.cutoff**2-x2)/(-x2))
    j0=.5*((q-x1)*l1/(x1-x2)+(q-x2)*l2/(x2-x1))
    jz=odd*(l1-l2)/(2*(x1-x2))
    r1=-np.arctanh(p.cutoff/np.sqrt(x1))/np.sqrt(x1)
    r2=-np.arctanh(p.cutoff/np.sqrt(x2))/np.sqrt(x2)
    jx=p.soc*(x1*r1-x2*r2)/(x1-x2)
    out=p.impurity_hybridization**2/(2*np.pi)*(j0[...,None,None]*I+jx[...,None,None]*X+jz[...,None,None]*Z)
    # Far from the band use a direct Gaussian radial integral, avoiding
    # cancellation in the logarithmic expression. This is the same kernel.
    far=np.abs(z)>2
    if np.any(far):
        nodes,weights=leggauss(32)
        k=(nodes+1)*p.cutoff/2
        weights=weights*p.cutoff/2*k/(2*np.pi)*p.impurity_hybridization**2
        kk=(q[far,None]-k[None,:]**2)[...,None,None]*I-p.soc*k[None,:,None,None]*X-odd[far,None,None,None]*Z
        out[far]=np.einsum('k,wkij->wij',weights,inv2(kk))
    return out

def matrices(prob,theta):
    e,pp,pm,d=prob
    n1=pp+d;n2=pm+d
    r1=(np.sqrt(e*pp)+np.sqrt(pm*d))/np.sqrt(n1*(1-n1))
    r2=(np.sqrt(e*pm)+np.sqrt(pp*d))/np.sqrt(n2*(1-n2))
    direction=np.sin(theta)*X+np.cos(theta)*Z
    R=(r1+r2)/2*I+(r1-r2)/2*direction
    n=(n1+n2)/2*I+(n1-n2)/2*direction
    return R,n,np.array([r1,r2])

def unpack(x):
    prob=softmax(np.r_[0.,x[:3]])
    R,n,rv=matrices(prob,x[3])
    lam=np.einsum('a,aij->ij',x[4:7],PAULI)
    return prob,R,n,rv,lam

class Saddle:
    def __init__(self,mu=0.,order=20,p=P):
        self.mu=mu;self.order=order;self.p=p
        self.y,self.w,self.ymax=integration_grid(order)
        self.omega=mu+1j*self.y
        self._vo=None;self._delta=None

    def set_mu(self,mu):
        if mu!=self.mu:
            self.mu=mu;self.omega=mu+1j*self.y;self._vo=None

    def set_W(self,W):
        if W!=self.p.impurity_hybridization:
            self.p=replace(self.p,impurity_hybridization=W);self._vo=None

    def delta(self,vo):
        if vo!=self._vo:
            self._delta=hybridization_array(self.omega,vo,self.p)
            self._vo=vo
        return self._delta

    def qp(self,R,lam,vo):
        delta=self.delta(vo)
        G=inv2(self.omega[:,None,None]*I-lam-R@delta@R)
        # n=I/2+int_0^infty Re G(mu+iy)dy/pi.
        nq=I/2+np.einsum('w,wij->ij',self.w,G.real)
        nq-=(lam-self.mu*I)/(np.pi*self.ymax)
        B=np.einsum('w,wij->ij',self.w,(delta@R@G+G@R@delta).real)
        mass=self.p.impurity_hybridization**2*self.p.cutoff**2/(4*np.pi)
        B-=2*mass*R/(np.pi*self.ymax)
        return nq,B

    def residual(self,x,U,eps,vo,force_scale=.02):
        prob,R,n,rv,lam=unpack(x)
        nq,B=self.qp(R,lam,vo)
        # Three independent probability directions; e=1-p+-p--d.
        forces=[]
        for j in range(1,4):
            pc=prob.astype(complex); pc[0]-=1e-25j;pc[j]+=1e-25j
            Rc,nc,_=matrices(pc,x[3])
            dr=Rc.imag/1e-25;dn=nc.imag/1e-25
            forces.append(np.trace((eps*I-lam)@dn+B@dr).real+(U if j==3 else 0.))
        # Divide the angular equation by the spin-density amplitude. Without
        # this regularization, n+=n- is a singular polar coordinate and can
        # falsely pass stationarity with a nonzero transverse local force.
        ns=(prob[1]-prob[2])/2
        if abs(ns)>1e-7:
            ratio=(rv[0]-rv[1])/(2*ns)
        else:
            pc=prob.astype(complex);pc[1]+=0.5e-20j;pc[2]-=0.5e-20j
            rvc=matrices(pc,x[3])[2]
            ratio=(rvc[0]-rvc[1]).imag/1e-20
        perpendicular=np.cos(x[3])*X-np.sin(x[3])*Z
        forces.append(np.trace((B*ratio-lam)@perpendicular).real)
        return np.r_[np.array(forces)/force_scale,components(nq-n).real]

    def initial(self,eps=EPS0,vo=VO0):
        n,_=self.qp(I,eps*I,vo)
        nv,V=np.linalg.eigh(n)
        n1,n2=nv[1],nv[0]
        direction=components(n)
        theta=np.arctan2(direction[1],direction[2])
        d=n1*n2
        prob=np.array([(1-n1)*(1-n2),n1-d,n2-d,d])
        return np.r_[np.log(prob[1:]/prob[0]),theta,eps,0,0]

    def solve(self,U,eps,vo,x0=None):
        if x0 is None:x0=self.initial(eps,vo)
        fun=lambda x:self.residual(x,U,eps,vo)
        sol=root(fun,x0,jac=lambda x:approx_derivative(fun,x,method='3-point',abs_step=1e-6),tol=1e-10)
        out=self.describe(sol.x,U,eps,vo)
        out.update(success=bool(np.max(np.abs(self.residual(sol.x,U,eps,vo)))<1e-8),solver_success=bool(sol.success),message=str(sol.message),nfev=int(sol.nfev))
        return out

    def describe(self,x,U,eps,vo):
        prob,R,n,rv,lam=unpack(x)
        nq,B=self.qp(R,lam,vo)
        return dict(U=float(U),epsilon=float(eps),auxiliary_vertex=float(vo),impurity_hybridization=self.p.impurity_hybridization,mu=float(self.mu),order=int(self.order),x=np.array(x).tolist(),probabilities=prob.tolist(),n_d=float(np.trace(n)),double_occupancy=float(prob[3]),density_eigenvalues=np.linalg.eigvalsh(n).tolist(),polarization=components(n)[1:].tolist(),Z_eigenvalues=(rv**2).tolist(),R=R.tolist(),lambda_matrix=lam.tolist(),constraint_max=float(np.max(abs(nq-n))),residual_max=float(np.max(abs(self.residual(x,U,eps,vo)))))

def kernel(z,x,vo,p=P):
    _,R,_,_,lam=unpack(x)
    return z*I-lam-R@bath.resonance_self_energy(z,vo,p)@R

def determinant(z,x,vo,p=P):
    return complex(np.linalg.det(kernel(z,x,vo,p)))

def ddet(z,x,vo,p=P,h=2e-6):
    return bath.derivative(lambda zz:determinant(zz,x,vo,p),z,h)

def joint_ep(saddle,U,y0,target_filling=None,control='V_O'):
    def residual(y):
        if target_filling is not None:saddle.set_mu(y[11]*.01)
        if control=='W':saddle.set_W(y[10]*.1)
        x=y[:7];z=(y[7]+1j*y[8])*.01;eps=y[9]*.01;vo=y[10] if control=='V_O' else VO0
        F=determinant(z,x,vo,saddle.p)
        Fz=ddet(z,x,vo,saddle.p)
        res=np.r_[saddle.residual(x,U,eps,vo),F.real/1e-5,F.imag/1e-5,Fz.real/.01,Fz.imag/.01]
        if target_filling is not None:res=np.r_[res,np.trace(unpack(x)[2])-target_filling]
        return res
    # A probability log ratio is exactly zero at n=1. Relative machine-sized
    # differencing there can generate a numerically zero Jacobian column.
    # Use a finite absolute central-difference step for every coordinate.
    sol=root(residual,y0,jac=lambda y:approx_derivative(residual,y,method='3-point',abs_step=1e-6),tol=2e-9)
    y=sol.x;x=y[:7];z=(y[7]+1j*y[8])*.01;eps=y[9]*.01;vo=y[10] if control=='V_O' else VO0
    if target_filling is not None:saddle.set_mu(y[11]*.01)
    if control=='W':saddle.set_W(y[10]*.1)
    out=saddle.describe(x,U,eps,vo)
    F=determinant(z,x,vo,saddle.p);Fz=ddet(z,x,vo,saddle.p)
    Fzz=bath.derivative(lambda zz:ddet(zz,x,vo,saddle.p),z,2e-6)
    out.update(y=y.tolist(),z=[z.real,z.imag],F=abs(F),F_z=abs(Fz),F_zz=[Fzz.real,Fzz.imag],singular_values=np.linalg.svd(kernel(z,x,vo,saddle.p),compute_uv=False).tolist(),open_channels=bath.open_channels(z.real,vo,saddle.p),joint_residual_max=float(np.max(abs(residual(y)))),solver_success=bool(sol.success),nfev=int(sol.nfev),message=str(sol.message))
    out['success']=bool(out['joint_residual_max']<1e-9 and z.imag<0 and len(out['open_channels'])==2 and min(out['probabilities'])>0 and vo>0 and saddle.p.impurity_hybridization>0)
    out['target_filling']=target_filling
    out['retuned_control']=control
    return out

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--mu',type=float,default=0.);ap.add_argument('--order',type=int,default=20);ap.add_argument('--out',default='results/smoke.json');ap.add_argument('--ep',action='store_true')
    a=ap.parse_args();s=Saddle(a.mu,a.order);rows=[];x=s.initial();y=np.r_[x,ZEP0.real/.01,ZEP0.imag/.01,EPS0/.01,VO0]
    for U in [0.,.0001,.0005,.001,.0025,.005,.01,.02,.04,.08,.12,.2]:
        row=joint_ep(s,U,y) if a.ep else s.solve(U,EPS0,VO0,x)
        rows.append(row)
        print(json.dumps({k:row[k] for k in ['U','success','n_d','double_occupancy','Z_eigenvalues','residual_max']}) ,flush=True)
        if not row['success']:break
        x=np.array(row['x']);y=np.array(row.get('y',y))
        Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(rows,indent=2))
    Path(a.out).write_text(json.dumps(rows,indent=2))
